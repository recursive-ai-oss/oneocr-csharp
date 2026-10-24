using OneImage;
using OneOcr.Detection;

namespace OneOcr.Layout;

/// <summary>
/// The orientation decision when a page detector is loaded: the detector's orientation head, acted on only when it is confident
/// (<see cref="PageOrientation.QuarterTurnsClockwise"/>). The reading it makes of the page is left in the evidence, so a page that
/// needs no turn is not detected a second time. A turn the head asks for is checked by the classical decision, which reads sample
/// lines both ways up: the head is trained on flat pages and is sure of itself on tilted photographs it has never seen (an upright
/// phone photo on a dark table, called upside down at 0.99), while a wrong turn costs the whole page.
/// </summary>
internal sealed class HeadOrientation : IOrientationDetector
{
    private readonly IPageDetector _detector;
    private readonly BufferPool<Gray8> _pixels;
    private readonly ResizerPool _resizers;
    private readonly IOrientationDetector? _check;

    public HeadOrientation(IPageDetector detector, BufferPool<Gray8> pixels, ResizerPool resizers, IOrientationDetector? check = null)
    {
        _check = check;
        _detector = detector;
        _pixels = pixels;
        _resizers = resizers;
    }

    /// <summary>The detector's reading of a page, with no decision made on it.</summary>
    public PageDetection Look(Gray8[] pixels, int width, int height, int degree, CancellationToken cancellationToken)
    {
        long t = StageClock.Start();
        PageDetection detection = _detector.Detect(pixels.AsSpan(0, checked(width * height)), width, height, _pixels, _resizers, degree, cancellationToken);
        StageClock.Stop("6 detect on straightened", t);
        return detection;
    }

    public int Detect(OrientationInput page, CancellationToken cancellationToken)
    {
        long t = StageClock.Start();
        PageDetection detection = _detector.Detect(
            page.Pixels.AsSpan(0, checked(page.Width * page.Height)), page.Width, page.Height, _pixels, _resizers, page.Degree, cancellationToken);
        StageClock.Stop("2 head detect", t);
        if (page.Evidence is { } evidence)
        {
            evidence.Detection = detection;
        }

        int turn = PageOrientation.QuarterTurnsClockwise(detection) * 90;
        if (turn == 0 || _check is null)
        {
            return turn;
        }

        if (page.Evidence is { } seen)
        {
            seen.PixelsMayHaveChanged = true;
        }

        t = StageClock.Start();
        int confirmed = _check.Detect(page, cancellationToken);
        StageClock.Stop("3 classical confirm", t);
        return confirmed;
    }
}
