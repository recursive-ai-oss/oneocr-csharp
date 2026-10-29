using OneOcr.Model;

namespace OneOcr.Layout;

/// <summary>
/// Replaces the confidences of a finished page by the model's reporting curve. The noise filter, the word joiner, the angle search
/// and the table tests read the confidence the model calibrates for decisions; a curve fitted for how well a reported number predicts
/// correctness would move those decisions and change the text. So the reporting curve is applied last, to the raw confidence, and
/// nothing reads its result.
/// </summary>
internal static class ReportedConfidence
{
    /// <summary>The page with each word's confidence mapped from its raw confidence, and line and block confidences recomputed the way the reader computes them.</summary>
    public static OcrPage Apply(OcrPage page, WordCalibration curve)
    {
        var blocks = new List<OcrBlock>(page.Blocks.Count);
        foreach (OcrBlock block in page.Blocks)
        {
            var lines = new List<OcrLine>(block.Lines.Count);
            double blockWeighted = 0;
            int blockWords = 0;
            foreach (OcrLine line in block.Lines)
            {
                var words = new List<OcrWord>(line.Words.Count);
                double weighted = 0, chars = 0;
                foreach (OcrWord word in line.Words)
                {
                    OcrWord mapped = word.Source == OcrSource.Ocr
                        ? word with { Confidence = curve.Apply(ScriptDetector.Of(word.Text), word.RawConfidence) }
                        : word;
                    words.Add(mapped);
                    weighted += mapped.Confidence * (double)mapped.Text.Length;
                    chars += mapped.Text.Length;
                }

                OcrLine made = line.Source == OcrSource.Ocr && chars > 0 ? line with { Words = words, Confidence = (float)(weighted / chars) } : line with { Words = words };
                lines.Add(made);
                blockWeighted += made.Confidence * (double)made.Words.Count;
                blockWords += made.Words.Count;
            }

            blocks.Add(block.Source == OcrSource.Ocr && blockWords > 0 ? block with { Lines = lines, Confidence = (float)(blockWeighted / blockWords) } : block with { Lines = lines });
        }

        return new OcrPage(
            page.Index, page.Width, page.Height, page.UnitsPerInch, blocks, page.Regions, page.Tables, page.KeyValues, page.Diagnostics, page.ToPdf, page.Orientation, page.SkewDegrees);
    }
}
