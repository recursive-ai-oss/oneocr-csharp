// Reads hello.png (two lines of text rendered with OneImage.Text, DejaVu Sans 44 px) with the bundled models, makes a searchable PDF
// from it and checks the PDF with OnePdf's text extraction, then reads fax.tif, a standard-mode fax (200 x 100 dots per inch, 1-bit).
using OneImage;
using OneOcr;
using OneOcr.Pdf;
using OnePdf;
using OnePdf.Text;

OcrEngine.ConfigureDefault();

string dir = AppContext.BaseDirectory;
Image image = Image.Load(Path.Combine(dir, "hello.png"));
string text = image.ToText();
Console.WriteLine(text);

OcrPage page = image.Ocr();
int words = page.Blocks.Sum(b => b.Lines.Sum(l => l.Words.Count));
Console.WriteLine($"{words} words");

bool ok = text.Contains("Invoice total 427.50", StringComparison.Ordinal)
    && text.Contains("Thank you for your order", StringComparison.Ordinal)
    && words == 8
    && image.ToText() == text;
Console.WriteLine(ok ? "image ok" : "image FAILED");

// A searchable PDF from the same image: OnePdf's extraction must return the words the page model has, in order.
using var pdf = new MemoryStream();
SearchablePdfResult made = image.ToSearchablePdf(pdf);
using PdfDocument doc = PdfDocument.Open(pdf.ToArray());
string pdfText = doc.Pages.Count == 1 ? doc.Pages[0].ExtractText().Text : "";
string[] expected = page.Words.Select(w => w.Text).ToArray();
string[] found = pdfText.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
bool pdfOk = made.Pages.Count == 1 && made.WordsWritten == words && found.SequenceEqual(expected);
Console.WriteLine(pdfOk ? "searchable pdf ok" : $"searchable pdf FAILED: '{pdfText}'");

// A standard-mode fax: the file declares 200 x 100 dpi, so the reader stretches it to square pixels before reading.
OcrPage fax = OcrEngine.Default.OcrPages(Path.Combine(dir, "fax.tif"))[0];
Console.WriteLine(fax.Text);
bool faxOk = fax.UnitsPerInch == 200 && fax.VerticalUnitsPerInch == 100
    && fax.Text.Contains("1,234.56", StringComparison.Ordinal)
    && fax.Text.Contains("2,500.00", StringComparison.Ordinal)
    && fax.Text.Contains("-312.75", StringComparison.Ordinal)
    && fax.Text.Contains("2,421.81", StringComparison.Ordinal);
Console.WriteLine(faxOk ? "fax ok" : "fax FAILED");

ok = ok && pdfOk && faxOk;
Console.WriteLine(ok ? "smoke ok" : "smoke FAILED");
return ok ? 0 : 1;
