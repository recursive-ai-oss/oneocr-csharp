// Renders planned text lines to model-height grayscale PNG crops with OneImage's text drawing.
//
// Input: a plan (JSONL, one line spec each: text, font file, size, colours, padding, degradation, seed).
// Output: <out>/images/<id>.png (8-bit gray, 32 px high) and <out>/labels.jsonl (id, text, size, PNG SHA-256).
// Rendering is deterministic: the same plan and font files give byte-identical PNGs.
//
// Pipeline per line: draw at the planned pixel size on a paper-coloured canvas, crop around the layout box
// with the planned padding, degrade at source resolution (blur, noise, JPEG), then scale to the model input
// height with Image.Resize(height: 32), the call the library's own preprocessing uses.

using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using OneImage;
using OneImage.Text;

string? planPath = null, outDir = null, fontRoot = Environment.GetEnvironmentVariable("ONEOCR_FONT_ROOT") ?? "/usr/share/fonts/truetype";
int height = 32;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--plan": planPath = args[++i]; break;
        case "--out": outDir = args[++i]; break;
        case "--font-root": fontRoot = args[++i]; break;
        case "--height": height = int.Parse(args[++i]); break;
        default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
    }
}

if (planPath is null || outDir is null)
{
    Console.Error.WriteLine("usage: LineGen --plan plan.jsonl --out dir [--font-root dir] [--height 32]");
    return 2;
}

Directory.CreateDirectory(Path.Combine(outDir, "images"));
var plan = File.ReadLines(planPath).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<PlanLine>(l, PlanJson.Options)!).ToArray();
var fontCache = new System.Collections.Concurrent.ConcurrentDictionary<string, FontFamily>();
var results = new string[plan.Length];

Parallel.For(0, plan.Length, new ParallelOptions { MaxDegreeOfParallelism = Math.Min(16, Environment.ProcessorCount) }, idx =>
{
    var p = plan[idx];
    FontFamily family = fontCache.GetOrAdd(p.Font, f => new FontCollection().Add(Path.Combine(fontRoot, f)));
    Image img = Render(p, family, height);
    string file = Path.Combine(outDir, "images", p.Id + ".png");
    img.Save(file);
    string sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file))).ToLowerInvariant();
    results[idx] = JsonSerializer.Serialize(new LabelLine(p.Id, p.Text, img.Width, img.Height, p.Font, sha), PlanJson.Options);
});

File.WriteAllLines(Path.Combine(outDir, "labels.jsonl"), results, new UTF8Encoding(false));
Console.WriteLine($"{plan.Length} lines -> {outDir}");
return 0;

static Image Render(PlanLine p, FontFamily family, int height)
{
    float em = (float)p.Em;
    Font font = family.CreateFont(em);
    var measure = new TextOptions(font);
    float advance = TextMeasurer.Measure(p.Text, measure).Advance.Width;

    // Canvas big enough for the largest padding the plan can ask for.
    float ox = 0.95f * em + 2, oy = 0.4f * em + 2;
    int w = (int)MathF.Ceiling(advance + 2 * ox), h = (int)MathF.Ceiling(em + 2 * oy);
    var bg = new Rgba32((byte)p.Bg, (byte)p.Bg, (byte)p.Bg, 255);
    var pixels = new Rgba32[w * h];
    Array.Fill(pixels, bg);
    var options = new TextOptions(font) { Origin = new Vector2(ox, oy) };
    Image canvas = Image.FromPixels<Rgba32>(pixels, w, h)
        .DrawText(p.Text, options, new Rgba32((byte)p.Fg, (byte)p.Fg, (byte)p.Fg, 255));

    int x0 = Math.Max(0, (int)MathF.Floor(ox - (float)p.Pad[0] * em));
    int x1 = Math.Min(w, (int)MathF.Ceiling(ox + advance + (float)p.Pad[1] * em));
    int y0 = Math.Max(0, (int)MathF.Floor(oy - (float)p.Pad[2] * em));
    int y1 = Math.Min(h, (int)MathF.Ceiling(oy + em + (float)p.Pad[3] * em));
    Image crop = canvas.ConvertTo(PixelFormat.Gray8).Crop(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));

    if (p.Blur > 0.05)
    {
        crop = crop.GaussianBlur(p.Blur);
    }

    if (p.Noise > 0.05)
    {
        crop = AddNoise(crop, p.Noise, (ulong)p.Seed);
    }

    if (p.Jpeg > 0)
    {
        using var ms = new MemoryStream();
        crop.Save(ms, JpegFormat.Instance, new SaveOptions { EncoderOptions = new JpegEncoderOptions { Quality = p.Jpeg } });
        ms.Position = 0;
        crop = Image.Load(ms);
    }

    // The model input scaling: width follows the aspect ratio (rounded half up, at least 1).
    return crop.Resize(height: height);
}

static Image AddNoise(Image img, double sigma, ulong seed)
{
    Gray8[] px = img.ToArray<Gray8>();
    ulong s = seed * 0x9E3779B97F4A7C15UL + 1;
    for (int i = 0; i < px.Length; i++)
    {
        double u1 = (Next(ref s) + 1.0) / 4294967297.0, u2 = Next(ref s) / 4294967296.0;
        double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        px[i] = new Gray8((byte)Math.Clamp((int)Math.Round(px[i].V + g * sigma), 0, 255));
    }

    return Image.FromPixels<Gray8>(px, img.Width, img.Height);
}

// SplitMix64, 32 bits per call.
static uint Next(ref ulong s)
{
    ulong z = s += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return (uint)((z ^ (z >> 31)) >> 32);
}

record PlanLine(string Id, string Text, string Font, double Em, int Fg, int Bg, double[] Pad, double Blur, double Noise, int Jpeg, long Seed);

record LabelLine(string Id, string Text, int W, int H, string Font, string Sha256);

static class PlanJson
{
    // Relaxed escaping keeps non-ASCII text readable in labels.jsonl.
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
}
