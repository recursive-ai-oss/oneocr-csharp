using OneImage;
using OneOcr.Detection;

namespace OneOcr.Layout;

/// <summary>
/// The orientation decision when a page detector is loaded: the detector's orientation head, acted on only when it is confident
/// (<see cref="PageOrientation.QuarterTurnsClockwise"/>). The reading it makes of the page is left in the evidence, so a page that
/// needs no turn is not detected a second time.
/// </summary>
internal sealed class HeadOrientation : IOrientationDetector
{
    private readonly IPageDetector _detector;
    private readonly BufferPool<Gray8> _pixels;
    private readonly ResizerPool _resizers;

    public HeadOrientation(IPageDetector detector, BufferPool<Gray8> pixels, ResizerPool resizers)
    {
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

        return PageOrientation.QuarterTurnsClockwise(detection) * 90;
    }
}
