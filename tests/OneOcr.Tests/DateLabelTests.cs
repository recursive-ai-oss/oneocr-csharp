using OneOcr.Documents;

namespace OneOcr.Tests;

public class DateLabelTests
{
    private static readonly DocSchema Schema = new(["BT-2", "BT-9", "BT-72"], ["none"], ["invoice"]);

    // One row: label words then the value, each 40 wide with a gap of 4.
    private static (string[] Texts, OcrBox[] Boxes) Row(params string[] words)
    {
        var boxes = new OcrBox[words.Length];
        for (int i = 0; i < words.Length; i++)
        {
            boxes[i] = new OcrBox(10 + (i * 44), 100, 50 + (i * 44), 111);
        }

        return (words, boxes);
    }

    [Theory]
    [InlineData("Due", "date:", 2, 1)] // the model said delivery, the label says due
    [InlineData("Fälligkeitsdatum:", "x", 2, 1)]
    [InlineData("Delivery", "date:", 1, 2)] // the model said due, the label says delivery
    [InlineData("Lieferdatum:", "x", 1, 2)]
    [InlineData("Date", "of", 2, 2)] // no due or delivery word: the model's tag stays
    [InlineData("Due", "date:", 1, 1)]
    public void The_printed_label_decides_between_due_and_delivery_date(string a, string b, int modelClass, int expected)
    {
        (string[] texts, OcrBox[] boxes) = Row(a, b, "26.02.2026");
        Assert.Equal(expected, DateLabel.Retag(Schema, modelClass, [2], texts, boxes));
    }

    [Fact]
    public void Other_classes_and_values_without_a_label_row_are_left_alone()
    {
        (string[] texts, OcrBox[] boxes) = Row("Due", "date:", "26.02.2026");
        Assert.Equal(0, DateLabel.Retag(Schema, 0, [2], texts, boxes));
        boxes[0] = new OcrBox(10, 300, 50, 311); // the label is on another row
        boxes[1] = new OcrBox(54, 300, 94, 311);
        Assert.Equal(2, DateLabel.Retag(Schema, 2, [2], texts, boxes));
    }
}
