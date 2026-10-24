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

    public int Detect(OrientationInput page, CancellationToken cancellationToken)
    {
        PageDetection detection = _detector.Detect(
            page.Pixels.AsSpan(0, checked(page.Width * page.Height)), page.Width, page.Height, _pixels, _resizers, page.Degree, cancellationToken);
        if (page.Evidence is { } evidence)
        {
            evidence.Detection = detection;
        }

        int turn = PageOrientation.QuarterTurnsClockwise(detection) * 90;
        return turn == 0 || _check is null ? turn : _check.Detect(page, cancellationToken);
    }
}
