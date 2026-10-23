namespace OneOcr;

/// <summary>
/// One word read from an image. Coordinates are pixels of the source image, origin top-left, y down.
/// </summary>
/// <param name="Text">The word's text; never empty, never containing the model's space characters.</param>
/// <param name="Left">Left edge.</param>
/// <param name="Top">Top edge: the top of the text line the word is on.</param>
/// <param name="Right">Right edge.</param>
/// <param name="Bottom">Bottom edge: the bottom of the text line the word is on.</param>
/// <param name="Confidence">
/// Calibrated confidence in 0..1 that the word is exactly right, from the model's per-script calibration curve. Equal to
/// <paramref name="RawConfidence"/> when the model file carries no calibration.
/// </param>
/// <param name="RawConfidence">
/// Mean over the word's characters of the largest softmax probability seen while that character was the winning class.
/// </param>
/// <param name="BoxIsApproximate">
/// <see langword="true"/> for every word the recognizer reads: the horizontal edges come from where the network's output
/// fires, which is only roughly where the glyphs are, and the vertical edges are those of the whole text line.
/// </param>
public readonly record struct OcrWord(
    string Text, float Left, float Top, float Right, float Bottom, float Confidence, float RawConfidence, bool BoxIsApproximate = true);

/// <summary>One text line read from an image: its words and the line's own confidence.</summary>
/// <param name="Text">The recognized text as the network produced it, including its own spacing.</param>
/// <param name="Words">The line's words, left to right. Empty when the line has no non-space text.</param>
/// <param name="Left">Left edge of the line's ink, in source-image pixels.</param>
/// <param name="Top">Top edge of the line's ink.</param>
/// <param name="Right">Right edge of the line's ink.</param>
/// <param name="Bottom">Bottom edge of the line's ink.</param>
/// <param name="Confidence">
/// The words' calibrated confidences averaged with each word weighted by its character count; 0 for a line without words.
/// </param>
public sealed record OcrLine(
    string Text, IReadOnlyList<OcrWord> Words, float Left, float Top, float Right, float Bottom, float Confidence);

/// <summary>
/// What <see cref="OcrEngine.Recognize"/> found on a page: lines top to bottom, each with words, boxes and confidences. This is
/// the recognizer's own result, deliberately small; it is not the full page model.
/// </summary>
/// <param name="Lines">Lines top to bottom, including lines that read as empty text (they have no words).</param>
/// <param name="Diagnostics">Notes for the user, for example that the model file has no calibration. Empty when there is nothing to say.</param>
public sealed record OcrRecognition(IReadOnlyList<OcrLine> Lines, IReadOnlyList<string> Diagnostics)
{
    /// <summary>
    /// The page text: non-empty lines joined with <c>\n</c>, the same as <see cref="OcrEngine.ToText(OneImage.Image, int, CancellationToken)"/>
    /// returns with greedy decoding.
    /// </summary>
    public string Text => string.Join('\n', Lines.Where(l => l.Text.Length > 0).Select(l => l.Text));
}
