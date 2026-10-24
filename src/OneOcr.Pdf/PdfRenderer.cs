using OneImage;
using OneImage.Pdf;

namespace OneOcr.Pdf;

/// <summary>
/// Draws the pages of one PDF through the OneImage PDF bridge. The document is loaded on first use, so a PDF whose pages need no
/// rendering is never rendered. Safe to share between threads: regions are drawn independently and identically at any degree.
/// </summary>
internal sealed class PdfRenderer
{
    private readonly Func<ReadOnlyMemory<byte>> _source;
    private readonly PdfOcrOptions _options;
    private readonly OcrLimits _limits;
    private readonly int _firstPage;
    private readonly int _pageCount;
    private readonly object _gate = new();
    private Image? _image;
    private Exception? _failure;
    private int _active;

    public PdfRenderer(Func<ReadOnlyMemory<byte>> source, PdfOcrOptions options, OcrLimits limits, int firstPage, int pageCount)
    {
        _limits = limits;
        _source = source;
        _options = options;
        _firstPage = firstPage;
        _pageCount = pageCount;
    }

    /// <summary>The frame that draws page <paramref name="pageIndex"/> at the render resolution.</summary>
    /// <exception cref="OneImage.ImageException">The PDF cannot be drawn; the failure is remembered, so every page reports it.</exception>
    public ImageFrame Frame(int pageIndex)
    {
        Image image;
        lock (_gate)
        {
            if (_failure is not null)
            {
                throw new ImageFormatException(_failure.Message, _failure);
            }

            if (_image is null)
            {
                try
                {
                    var render = new PdfRenderOptions
                    {
                        Dpi = _options.RenderDpi,
                        FirstPage = _firstPage,
                        PageCount = _pageCount,
                        Password = _options.Password,
                    };
                    var configuration = new ImageConfiguration().Add(new PdfFormat(render));
                    _image = Image.Load(_source(), new LoadOptions
                    {
                        Configuration = configuration,
                        // A page larger than the recognizer accepts is refused before anything is drawn.
                        Limits = new ImageLimits { MaxPixelsPerFrame = _limits.MaxImagePixels, MaxFrames = Math.Max(1, _pageCount) },
                    });
                }
                catch (ImageException e)
                {
                    _failure = e;
                    throw;
                }
                catch (OnePdf.PdfException e)
                {
                    _failure = e;
                    throw new ImageFormatException(e.Message, e);
                }
            }

            image = _image;
        }

        int slot = pageIndex - _firstPage;
        if (slot < 0 || slot >= image.Frames.Count)
        {
            throw new ImageFormatException("The page is not among the rendered pages.");
        }

        lock (_gate)
        {
            _active++;
        }

        return image.Frames[slot];
    }

    /// <summary>
    /// A page obtained from <see cref="Frame"/> has been read. The image keeps the pixels of every region read from a page (about
    /// a byte per pixel of the page) until they are released, which would add up over a long document; they are given back as soon
    /// as no page is being read. (Releasing takes the whole image, so it must not happen while another page reads.)
    /// </summary>
    public void Done()
    {
        lock (_gate)
        {
            if (--_active == 0)
            {
                _image?.ReleaseDecodedPixels();
            }
        }
    }
}
