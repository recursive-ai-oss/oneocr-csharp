using System;
using System.Linq;
using OneOcr;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Tests;

// Builds a small random-weight recognizer, runs it, and checks the failure paths, all under NativeAOT.
var shape = new TestModels.Shape(8, [(4, 2, 2), (4, 2, 2), (8, 2, 1)], 8, 2, 12);
byte[] file = TestModels.RandomContainer(shape, 1);
LineRecognizer recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(file));
byte[] image = TestModels.RandomImage(8, 96, 2);
RecognizedLine a = recognizer.Recognize(image, 96, 1);
RecognizedLine b = recognizer.Recognize(image, 96, 4);
if (a != b)
{
    Console.Error.WriteLine("degrees disagree");
    return 1;
}

try
{
    ModelContainer.Parse(file.AsSpan(0, file.Length - 1));
    Console.Error.WriteLine("truncated file was accepted");
    return 1;
}
catch (OcrModelException)
{
}

// A page of four text-like lines through the whole pipeline: line finding, scaling, recognition, and the default engine.
var pageShape = new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41);
OcrEngine engine = OcrEngine.Load(new System.IO.MemoryStream(TestModels.RandomContainer(pageShape, 3)));
var pixels = new OneImage.Gray8[600 * 300];
Array.Fill(pixels, new OneImage.Gray8(255));
foreach (int top in new[] { 40, 110, 180 })
{
    for (int y = top; y < top + 30; y++)
    {
        for (int x = 60; x < 500; x++)
        {
            if ((x / 7) % 3 != 0)
            {
                pixels[(y * 600) + x] = new OneImage.Gray8(10);
            }
        }
    }
}

OneImage.Image page = OneImage.Image.FromPixels<OneImage.Gray8>(pixels, 600, 300);
string t1 = engine.ToText(page, 1), t4 = page.ToText(engine);
if (t1 != t4 || t1.Split('\n').Length != 3)
{
    Console.Error.WriteLine("ToText disagrees or found the wrong number of lines");
    return 1;
}

// Beam search with a language model and a lexicon: same text at every degree of parallelism, and a model that fits the charset.
string[] charset = TestModels.Charset(41);
string[] alphabet = [.. charset.Skip(2).Take(12)];
var beam = new OneOcr.Decoding.DecodeOptions
{
    Mode = OneOcr.Decoding.CtcDecoding.Beam,
    BeamWidth = 4,
    LanguageModel = OneOcr.Decoding.CharLanguageModel.Parse(TestLm.Build("smoke", alphabet, [string.Concat(alphabet.Take(4)), string.Concat(alphabet.Skip(2).Take(5))], 3)),
    Lexicon = OneOcr.Decoding.Lexicon.FromWords([string.Concat(alphabet.Take(3)), "x1"]),
};
string b1 = engine.ToText(page, beam, 1), b4 = engine.WithDecoding(beam).ToText(page, 4);
if (b1 != b4 || b1.Split('\n').Length != 3)
{
    Console.Error.WriteLine("beam decoding disagrees across degrees");
    return 1;
}

// A version 1 container: calibration is parsed (source-generated JSON), applied, and words come back with boxes.
OcrEngine calibrated = OcrEngine.Load(new System.IO.MemoryStream(TestModels.RandomContainerV1(pageShape, 3,
    "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,1],\"y\":[0,0.5]}}}")));
OneOcr.OcrPage words = calibrated.Ocr(page, new OneOcr.OcrOptions { MaxDegreeOfParallelism = 1 });
if (words.ToText() != t1 || words.Blocks.Sum(bl => bl.Lines.Count) != 3 || calibrated.Ocr(page, new OneOcr.OcrOptions { MaxDegreeOfParallelism = 4 }).ToText() != t1)
{
    Console.Error.WriteLine("Ocr disagrees with ToText");
    return 1;
}

// The page model's JSON: write, read, write again (source-generated, no reflection), and a malformed document is a typed error.
string json = OneOcr.OcrPageJson.ToJson([words]);
System.Collections.Generic.IReadOnlyList<OneOcr.OcrPage> back = OneOcr.OcrPageJson.Parse(json);
if (back.Count != 1 || OneOcr.OcrPageJson.ToJson(back) != json || back[0].ToText() != t1)
{
    Console.Error.WriteLine("page JSON does not round trip");
    return 1;
}

try
{
    OneOcr.OcrPageJson.Parse(json.Replace("\"oneocr.page\"", "\"other\""));
    Console.Error.WriteLine("wrong format accepted");
    return 1;
}
catch (OneOcr.OcrFormatException)
{
}

OneOcr.OneOcr.Configure(engine);
if (page.ToText() != t1)
{
    Console.Error.WriteLine("default engine disagrees");
    return 1;
}

Console.WriteLine($"ok: {a.Text.Length} characters, confidence {a.Confidence:F3}");
return 0;
