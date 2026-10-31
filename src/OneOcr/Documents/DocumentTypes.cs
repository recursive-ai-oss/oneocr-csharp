using System.Diagnostics.CodeAnalysis;

namespace OneOcr.Documents;

/// <summary>The kind of document the extraction model takes a page to be.</summary>
public enum DocumentType
{
    /// <summary>Something that is neither: no fields are read, only generic key–value pairs.</summary>
    Other = 0,

    /// <summary>An invoice.</summary>
    Invoice = 1,

    /// <summary>A receipt.</summary>
    Receipt = 2,
}

/// <summary>How the text of a field was normalized into <see cref="ExtractedField.Value"/>.</summary>
public enum FieldValueKind
{
    /// <summary>Kept as read.</summary>
    Text = 0,

    /// <summary>A calendar date, written <c>yyyy-MM-dd</c> (<c>--MM-dd</c> when the page gives no year).</summary>
    Date = 1,

    /// <summary>A time of day, kept as read.</summary>
    Time = 2,

    /// <summary>A money amount: no grouping, <c>.</c> as the decimal mark.</summary>
    Amount = 3,

    /// <summary>A quantity, written like an amount.</summary>
    Quantity = 4,

    /// <summary>A rate in percent (the sign is dropped), written like an amount.</summary>
    Percent = 5,

    /// <summary>An ISO 4217 currency code, read from a code or an unambiguous symbol.</summary>
    Currency = 6,
}

/// <summary>
/// One value the extraction model read: a run of words of the page. The text is exactly the words' text; the normalized value is derived
/// from it by the rules in <c>docs/spec/grounding.md</c>, the same ones the LLM layer uses to check values against the page.
/// Records compare by value (the word indices element by element).
/// </summary>
public sealed record ExtractedField
{
    /// <summary>Creates an empty field; set the required members.</summary>
    public ExtractedField()
    {
    }

    [SetsRequiredMembers]
    internal ExtractedField(
        string id, string text, IReadOnlyList<int> wordIndices, OcrBox box, float confidence, FieldValueKind kind, string value,
        decimal? number = null, DateOnly? date = null, string? currency = null)
    {
        Id = id;
        Text = text;
        WordIndices = wordIndices;
        Box = box;
        Confidence = confidence;
        Kind = kind;
        Value = value;
        Number = number;
        Date = date;
        Currency = currency;
    }

    /// <summary>The field's class: an EN 16931 business term such as <c>BT-1</c> for invoices, <c>receipt.total</c> for receipts, or a name a custom model defines.</summary>
    public required string Id { get; init; }

    /// <summary>The words joined by one space.</summary>
    public required string Text { get; init; }

    /// <summary>Indices into <see cref="OcrPage.Words"/>, in reading order.</summary>
    public required IReadOnlyList<int> WordIndices { get; init; }

    /// <summary>The union of the words' boxes, in page units.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>Mean probability of the words' chosen tags, 0 to 1. Not a calibrated probability of being right.</summary>
    public required float Confidence { get; init; }

    /// <summary>How <see cref="Value"/> was derived.</summary>
    public required FieldValueKind Kind { get; init; }

    /// <summary>The normalized value: equal to <see cref="Text"/> when the text could not be normalized or is plain text.</summary>
    public required string Value { get; init; }

    /// <summary>The amount, quantity or rate when <see cref="Kind"/> says so and the text parsed; otherwise null.</summary>
    public decimal? Number { get; init; }

    /// <summary>The date when it parsed unambiguously and includes a year; otherwise null.</summary>
    public DateOnly? Date { get; init; }

    /// <summary>The ISO code of the currency written with the amount, when it names exactly one; otherwise null.</summary>
    public string? Currency { get; init; }

    /// <inheritdoc />
    public bool Equals(ExtractedField? other) =>
        other is not null && Id == other.Id && Text == other.Text && Box == other.Box && Confidence.Equals(other.Confidence) && Kind == other.Kind
        && Value == other.Value && Number == other.Number && Date == other.Date && Currency == other.Currency && ModelEquality.Same(WordIndices, other.WordIndices);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Id, Text, Box, Confidence, Kind, Value, WordIndices.Count);
}

/// <summary>One repeated group of fields: a line item or a row of the tax breakdown. Each field class appears at most once. Records compare by value (the fields element by element).</summary>
public sealed record ExtractedGroup
{
    /// <summary>Creates an empty group; set the required members.</summary>
    public ExtractedGroup()
    {
    }

    [SetsRequiredMembers]
    internal ExtractedGroup(int anchor, IReadOnlyList<ExtractedField> fields)
    {
        Anchor = anchor;
        Fields = fields;
    }

    /// <summary>Index into <see cref="OcrPage.Words"/> of the word the group's other words point to (its first word).</summary>
    public required int Anchor { get; init; }

    /// <summary>The fields, in reading order.</summary>
    public required IReadOnlyList<ExtractedField> Fields { get; init; }

    /// <summary>The group's field of a class, or null when the group has none.</summary>
    /// <param name="id">The class, for example <c>BT-129</c>.</param>
    /// <returns>The field or null.</returns>
    public ExtractedField? Get(string id) => Fields.FirstOrDefault(f => f.Id == id);

    /// <inheritdoc />
    public bool Equals(ExtractedGroup? other) => other is not null && Anchor == other.Anchor && ModelEquality.Same(Fields, other.Fields);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Anchor, Fields.Count);
}

/// <summary>What the extraction model read from a page, before it is shaped into an invoice or a receipt.</summary>
public sealed class DocumentExtraction
{
    internal DocumentExtraction(
        DocumentType documentType, float documentTypeConfidence, IReadOnlyList<ExtractedField> fields, IReadOnlyList<ExtractedGroup> lineItems,
        IReadOnlyList<ExtractedGroup> taxRows, IReadOnlyList<OcrKeyValue> keyValues, IReadOnlyList<OcrDiagnostic> diagnostics)
    {
        DocumentType = documentType;
        DocumentTypeConfidence = documentTypeConfidence;
        Fields = fields;
        LineItems = lineItems;
        TaxRows = taxRows;
        KeyValues = keyValues;
        Diagnostics = diagnostics;
    }

    /// <summary>The kind of document the model takes the page to be, whatever schema was asked for.</summary>
    public DocumentType DocumentType { get; }

    /// <summary>Probability the model gives <see cref="DocumentType"/>, 0 to 1.</summary>
    public float DocumentTypeConfidence { get; }

    /// <summary>The once-per-page fields that were found, in class order; at most one per class (the most confident).</summary>
    public IReadOnlyList<ExtractedField> Fields { get; }

    /// <summary>Line items, ordered by their first word on the page.</summary>
    public IReadOnlyList<ExtractedGroup> LineItems { get; }

    /// <summary>Rows of the tax breakdown (invoices), in the same order.</summary>
    public IReadOnlyList<ExtractedGroup> TaxRows { get; }

    /// <summary>Generic key–value pairs found on the page, whatever its type; word indices refer to <see cref="OcrPage.Words"/>.</summary>
    public IReadOnlyList<OcrKeyValue> KeyValues { get; }

    /// <summary>Notes about the extraction, for example that the page has no words.</summary>
    public IReadOnlyList<OcrDiagnostic> Diagnostics { get; }

    /// <summary>The page-level field of a class.</summary>
    /// <param name="id">The class, for example <c>BT-1</c> or <c>receipt.total</c>.</param>
    /// <returns>The field, or null when none was found.</returns>
    public ExtractedField? Get(string id) => Fields.FirstOrDefault(f => f.Id == id);
}

/// <summary>An invoice read from a page. Every value is a span of the page's words; absent fields are null. BT numbers are EN 16931 business terms.</summary>
public sealed class Invoice
{
    internal Invoice(DocumentExtraction extraction)
    {
        Extraction = extraction;
        Lines = [.. extraction.LineItems.Select(static g => new InvoiceLine(g))];
        TaxBreakdown = [.. extraction.TaxRows.Select(static g => new InvoiceTaxRow(g))];
    }

    /// <summary>The underlying extraction, with the document type the model assigned and the generic key–value pairs.</summary>
    public DocumentExtraction Extraction { get; }

    /// <summary>BT-1, invoice number.</summary>
    public ExtractedField? Number => Extraction.Get("BT-1");

    /// <summary>BT-2, invoice issue date.</summary>
    public ExtractedField? IssueDate => Extraction.Get("BT-2");

    /// <summary>BT-5, invoice currency code.</summary>
    public ExtractedField? Currency => Extraction.Get("BT-5");

    /// <summary>BT-9, payment due date.</summary>
    public ExtractedField? DueDate => Extraction.Get("BT-9");

    /// <summary>BT-13, purchase order reference.</summary>
    public ExtractedField? PurchaseOrder => Extraction.Get("BT-13");

    /// <summary>BT-20, payment terms.</summary>
    public ExtractedField? PaymentTerms => Extraction.Get("BT-20");

    /// <summary>BT-27, seller name.</summary>
    public ExtractedField? SellerName => Extraction.Get("BT-27");

    /// <summary>BT-31, seller VAT identifier.</summary>
    public ExtractedField? SellerVatId => Extraction.Get("BT-31");

    /// <summary>BT-35, seller address line.</summary>
    public ExtractedField? SellerAddress => Extraction.Get("BT-35");

    /// <summary>BT-37, seller city.</summary>
    public ExtractedField? SellerCity => Extraction.Get("BT-37");

    /// <summary>BT-38, seller post code.</summary>
    public ExtractedField? SellerPostCode => Extraction.Get("BT-38");

    /// <summary>BT-40, seller country code.</summary>
    public ExtractedField? SellerCountry => Extraction.Get("BT-40");

    /// <summary>BT-44, buyer name.</summary>
    public ExtractedField? BuyerName => Extraction.Get("BT-44");

    /// <summary>BT-48, buyer VAT identifier.</summary>
    public ExtractedField? BuyerVatId => Extraction.Get("BT-48");

    /// <summary>BT-50, buyer address line.</summary>
    public ExtractedField? BuyerAddress => Extraction.Get("BT-50");

    /// <summary>BT-52, buyer city.</summary>
    public ExtractedField? BuyerCity => Extraction.Get("BT-52");

    /// <summary>BT-53, buyer post code.</summary>
    public ExtractedField? BuyerPostCode => Extraction.Get("BT-53");

    /// <summary>BT-55, buyer country code.</summary>
    public ExtractedField? BuyerCountry => Extraction.Get("BT-55");

    /// <summary>BT-72, actual delivery date.</summary>
    public ExtractedField? DeliveryDate => Extraction.Get("BT-72");

    /// <summary>BT-84, payment account identifier (IBAN).</summary>
    public ExtractedField? Iban => Extraction.Get("BT-84");

    /// <summary>BT-86, payment service provider identifier (BIC).</summary>
    public ExtractedField? Bic => Extraction.Get("BT-86");

    /// <summary>BT-106, sum of invoice line net amounts.</summary>
    public ExtractedField? Subtotal => Extraction.Get("BT-106");

    /// <summary>BT-110, invoice total VAT amount.</summary>
    public ExtractedField? TaxTotal => Extraction.Get("BT-110");

    /// <summary>BT-112, invoice total amount with VAT.</summary>
    public ExtractedField? Total => Extraction.Get("BT-112");

    /// <summary>BT-115, amount due for payment.</summary>
    public ExtractedField? AmountDue => Extraction.Get("BT-115");

    /// <summary>The invoice lines, in page order.</summary>
    public IReadOnlyList<InvoiceLine> Lines { get; }

    /// <summary>The VAT breakdown rows, in page order.</summary>
    public IReadOnlyList<InvoiceTaxRow> TaxBreakdown { get; }
}

/// <summary>One invoice line.</summary>
public sealed class InvoiceLine
{
    internal InvoiceLine(ExtractedGroup group) => Group = group;

    /// <summary>The line's fields as read.</summary>
    public ExtractedGroup Group { get; }

    /// <summary>BT-126, invoice line identifier.</summary>
    public ExtractedField? Id => Group.Get("BT-126");

    /// <summary>BT-153, item name (the line's description).</summary>
    public ExtractedField? Description => Group.Get("BT-153");

    /// <summary>BT-129, invoiced quantity.</summary>
    public ExtractedField? Quantity => Group.Get("BT-129");

    /// <summary>BT-130, invoiced quantity unit of measure.</summary>
    public ExtractedField? Unit => Group.Get("BT-130");

    /// <summary>BT-146, item net price.</summary>
    public ExtractedField? UnitPrice => Group.Get("BT-146");

    /// <summary>BT-131, invoice line net amount.</summary>
    public ExtractedField? Amount => Group.Get("BT-131");

    /// <summary>BT-152, invoiced item VAT rate.</summary>
    public ExtractedField? TaxRate => Group.Get("BT-152");
}

/// <summary>One row of the VAT breakdown.</summary>
public sealed class InvoiceTaxRow
{
    internal InvoiceTaxRow(ExtractedGroup group) => Group = group;

    /// <summary>The row's fields as read.</summary>
    public ExtractedGroup Group { get; }

    /// <summary>BT-116, VAT category taxable amount.</summary>
    public ExtractedField? TaxableAmount => Group.Get("BT-116");

    /// <summary>BT-117, VAT category tax amount.</summary>
    public ExtractedField? TaxAmount => Group.Get("BT-117");

    /// <summary>BT-119, VAT category rate.</summary>
    public ExtractedField? Rate => Group.Get("BT-119");
}

/// <summary>A receipt read from a page. Every value is a span of the page's words; absent fields are null.</summary>
public sealed class Receipt
{
    internal Receipt(DocumentExtraction extraction)
    {
        Extraction = extraction;
        Items = [.. extraction.LineItems.Select(static g => new ReceiptItem(g))];
    }

    /// <summary>The underlying extraction, with the document type the model assigned and the generic key–value pairs.</summary>
    public DocumentExtraction Extraction { get; }

    /// <summary>The merchant's name.</summary>
    public ExtractedField? Merchant => Extraction.Get("receipt.merchant");

    /// <summary>The date of the purchase.</summary>
    public ExtractedField? Date => Extraction.Get("receipt.date");

    /// <summary>The time of the purchase.</summary>
    public ExtractedField? Time => Extraction.Get("receipt.time");

    /// <summary>The subtotal before tax.</summary>
    public ExtractedField? Subtotal => Extraction.Get("receipt.subtotal");

    /// <summary>The tax.</summary>
    public ExtractedField? Tax => Extraction.Get("receipt.tax");

    /// <summary>The tip.</summary>
    public ExtractedField? Tip => Extraction.Get("receipt.tip");

    /// <summary>The total.</summary>
    public ExtractedField? Total => Extraction.Get("receipt.total");

    /// <summary>How it was paid.</summary>
    public ExtractedField? PaymentMethod => Extraction.Get("receipt.paymentMethod");

    /// <summary>The purchased items, in page order.</summary>
    public IReadOnlyList<ReceiptItem> Items { get; }
}

/// <summary>One item on a receipt.</summary>
public sealed class ReceiptItem
{
    internal ReceiptItem(ExtractedGroup group) => Group = group;

    /// <summary>The item's fields as read.</summary>
    public ExtractedGroup Group { get; }

    /// <summary>The item's name.</summary>
    public ExtractedField? Name => Group.Get("receipt.itemName");

    /// <summary>The quantity.</summary>
    public ExtractedField? Quantity => Group.Get("receipt.quantity");

    /// <summary>The price of one unit.</summary>
    public ExtractedField? UnitPrice => Group.Get("receipt.unitPrice");

    /// <summary>The amount charged for the item.</summary>
    public ExtractedField? Amount => Group.Get("receipt.amount");
}
