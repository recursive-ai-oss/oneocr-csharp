using System;
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

Console.WriteLine($"ok: {a.Text.Length} characters, confidence {a.Confidence:F3}");
return 0;
