// Reads hello.png (two lines of text rendered with OneImage.Text, DejaVu Sans 44 px) with the bundled models.
using OneImage;
using OneOcr;

OcrEngine.ConfigureDefault();

Image image = Image.Load(Path.Combine(AppContext.BaseDirectory, "hello.png"));
string text = image.ToText();
Console.WriteLine(text);

OcrPage page = image.Ocr();
int words = page.Blocks.Sum(b => b.Lines.Sum(l => l.Words.Count));
Console.WriteLine($"{words} words");

bool ok = text.Contains("Invoice total 427.50", StringComparison.Ordinal)
    && text.Contains("Thank you for your order", StringComparison.Ordinal)
    && words == 8
    && image.ToText() == text;
Console.WriteLine(ok ? "smoke ok" : "smoke FAILED");
return ok ? 0 : 1;
