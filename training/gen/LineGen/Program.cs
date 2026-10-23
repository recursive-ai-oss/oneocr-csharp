// Renders text lines to model-height grayscale PNG crops with OneImage's text drawing.
//
//   LineGen --plan plan.jsonl --out dir [--font-root dir] [--height 32]
//       Renders planned synthetic lines. Output: dir/images/<id>.png (8-bit gray) and dir/labels.jsonl.
//   LineGen --plan plan.jsonl --out dir --clean [--height 48]
//       Clean pack for the GPU degradation stage: black text on white at a fixed font size (the plan's size, colours and
//       degradations are ignored; the stage re-creates them), crops scaled to --height, all lines concatenated in
//       dir/clean.u8 (row-major H x W bytes per line) and described by dir/labels.jsonl (id, text, w, h, off, font, sha256).
//   LineGen --page-lines lines.jsonl --out dir [--height 32]
//       Cuts evaluation crops from page images. Each input line is {id, text, page, box:[x0,y0,x1,y1]} (truth box).
//       Output: dir/finder (crops cut the way the library's line finder does), dir/box (the truth box itself) and dir/margin
//       (the truth box with the library's margins).
//
// Rendering is deterministic: the same plan and fonts give byte-identical PNGs.
//
// Crop convention (shared by the synthetic lines and the page crops, so training sees what inference sees): the ink box of
// the line (ascender top to descender bottom) has height H; when H is below 0.85 x the page's median line height, H is raised
// to that (lines without descenders, digit-only lines). The crop is centred on the ink box, extends H/4 above and below it
// (1.5 H tall) and H/2 left and right of the ink, then is scaled to the model height with Image.Resize(height: 32).
// Synthetic lines jitter these margins (plan fields vm, hm) so the model tolerates a finder that is a little off.

using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using OneImage;
using OneImage.Text;

string? planPath = null, pageLinesPath = null, outDir = null, coverageList = null, charsetPath = null;
const double CleanEm = 28;  // font size of clean renders; the GPU stage supplies the apparent size
bool clean = false;
string fontRoot = Environment.GetEnvironmentVariable("ONEOCR_FONT_ROOT") ?? "fonts-cache";
int height = 32;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--plan": planPath = args[++i]; break;
        case "--page-lines": pageLinesPath = args[++i]; break;
        case "--out": outDir = args[++i]; break;
        case "--coverage": coverageList = args[++i]; break;
        case "--charset": charsetPath = args[++i]; break;
        case "--font-root": fontRoot = args[++i]; break;
        case "--clean": clean = true; break;
        case "--height": height = int.Parse(args[++i]); break;
        default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
    }
}

if (coverageList is not null && outDir is not null && charsetPath is not null)
{
    return RunCoverage(coverageList, charsetPath, outDir, fontRoot);
}

if (outDir is null || (planPath is null) == (pageLinesPath is null))
{
    Console.Error.WriteLine("usage: LineGen (--plan plan.jsonl | --page-lines lines.jsonl) --out dir [--font-root dir] [--height 32]");
    Console.Error.WriteLine("       LineGen --coverage fonts.txt --charset charset.txt --out coverage.json [--font-root dir]");
    return 2;
}

var po = new ParallelOptions { MaxDegreeOfParallelism = Math.Min(24, Environment.ProcessorCount) };
return planPath is not null ? RunPlan(planPath, outDir, fontRoot, height, po, clean) : RunPageLines(pageLinesPath!, outDir, height, po);

// For every font file in the list, the charset entries the renderer can actually draw (a cmap entry can still point at an empty glyph).
static int RunCoverage(string listPath, string charsetPath, string outFile, string fontRoot)
{
    string[] chars = File.ReadAllLines(charsetPath).Skip(1).ToArray();
    string[] files = File.ReadAllLines(listPath).Where(l => l.Length > 0).ToArray();
    var result = new SortedDictionary<string, string>(StringComparer.Ordinal);
    foreach (string f in files)
    {
        Font font = new FontCollection().Add(Path.Combine(fontRoot, f)).CreateFont(32);
        result[f] = string.Concat(chars.Where(c => font.HasGlyph(char.ConvertToUtf32(c, 0))));
    }

    File.WriteAllText(outFile, JsonSerializer.Serialize(result, PlanJson.Options), new UTF8Encoding(false));
    Console.WriteLine($"{files.Length} fonts -> {outFile}");
    return 0;
}

static int RunPlan(string planPath, string outDir, string fontRoot, int height, ParallelOptions po, bool clean)
{
    if (clean)
    {
        return RunClean(planPath, outDir, fontRoot, height, po);
    }

    Directory.CreateDirectory(Path.Combine(outDir, "images"));
    var plan = File.ReadLines(planPath).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<PlanLine>(l, PlanJson.Options)!).ToArray();
    var fontCache = new System.Collections.Concurrent.ConcurrentDictionary<string, FontFamily>();
    var results = new string[plan.Length];
    Parallel.For(0, plan.Length, po, idx =>
    {
        var p = plan[idx];
        FontFamily family = fontCache.GetOrAdd(p.Font, f => new FontCollection().Add(Path.Combine(fontRoot, f)));
        Image img = RenderSynthetic(p, family, height);
        results[idx] = Save(img, outDir, "images", p.Id, p.Text, p.Font);
    });
    File.WriteAllLines(Path.Combine(outDir, "labels.jsonl"), results, new UTF8Encoding(false));
    Console.WriteLine($"{plan.Length} lines -> {outDir}");
    return 0;
}

static int RunClean(string planPath, string outDir, string fontRoot, int height, ParallelOptions po)
{
    Directory.CreateDirectory(outDir);
    var plan = File.ReadLines(planPath).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<PlanLine>(l, PlanJson.Options)!).ToArray();
    var fontCache = new System.Collections.Concurrent.ConcurrentDictionary<string, FontFamily>();
    var labels = new List<string>(plan.Length);
    long off = 0;
    using var bin = new FileStream(Path.Combine(outDir, "clean.u8"), FileMode.Create, FileAccess.Write);
    const int Chunk = 20000;
    for (int s = 0; s < plan.Length; s += Chunk)
    {
        int n = Math.Min(Chunk, plan.Length - s);
        var rows = new byte[n][];
        var w = new int[n];
        Parallel.For(0, n, po, k =>
        {
            var p = plan[s + k] with { Em = CleanEm, Fg = 0, Bg = 255, Blur = 0, Noise = 0, Jpeg = 0 };
            FontFamily family = fontCache.GetOrAdd(p.Font, f => new FontCollection().Add(Path.Combine(fontRoot, f)));
            Image img = RenderSynthetic(p, family, height);
            w[k] = img.Width;
            rows[k] = img.ToArray<Gray8>().Select(g => g.V).ToArray();
        });
        for (int k = 0; k < n; k++)
        {
            var p = plan[s + k];
            bin.Write(rows[k]);
            string sha = Convert.ToHexString(SHA256.HashData(rows[k])).ToLowerInvariant();
            labels.Add(JsonSerializer.Serialize(new PackLine(p.Id, p.Text, w[k], height, off, p.Font, sha), PlanJson.Options));
            off += rows[k].Length;
        }
    }

    File.WriteAllLines(Path.Combine(outDir, "labels.jsonl"), labels, new UTF8Encoding(false));
    Console.WriteLine($"{plan.Length} clean lines -> {outDir} ({off} bytes)");
    return 0;
}

static string Save(Image img, string outDir, string sub, string id, string text, string font)
{
    string file = Path.Combine(outDir, sub, id + ".png");
    img.Save(file);
    string sha = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file))).ToLowerInvariant();
    return JsonSerializer.Serialize(new LabelLine(id, text, img.Width, img.Height, font, sha), PlanJson.Options);
}

static Image RenderSynthetic(PlanLine p, FontFamily family, int height)
{
    float em = (float)p.Em;
    Font font = family.CreateFont(em);
    var options = new TextOptions(font);
    float advance = TextMeasurer.Measure(p.Text, options).Advance.Width;

    // Reference height: ink extent of a string with ascenders and descenders, standing in for the page's median line.
    float refH = RefInkHeight(font);

    // The canvas leaves room for the largest margins the plan can ask for.
    float ox = 1.8f * em + 2, oy = 1.4f * em + 2;
    int w = (int)MathF.Ceiling(advance + 2 * ox), h = (int)MathF.Ceiling(em + 2 * oy);
    var bg = new Rgba32((byte)p.Bg, (byte)p.Bg, (byte)p.Bg, 255);
    var pixels = new Rgba32[w * h];
    Array.Fill(pixels, bg);
    options.Origin = new Vector2(ox, oy);
    Image canvas = Image.FromPixels<Rgba32>(pixels, w, h).DrawText(p.Text, options, new Rgba32((byte)p.Fg, (byte)p.Fg, (byte)p.Fg, 255));
    Image gray = canvas.ConvertTo(PixelFormat.Gray8);
    Gray8[] px = gray.ToArray<Gray8>();

    int thr = Math.Max(1, (int)(0.4 * Math.Abs(p.Fg - p.Bg)));
    int ix0 = w, iy0 = h, ix1 = -1, iy1 = -1;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (Math.Abs(px[y * w + x].V - p.Bg) >= thr)
            {
                if (x < ix0) ix0 = x;
                if (x > ix1) ix1 = x;
                if (y < iy0) iy0 = y;
                if (y > iy1) iy1 = y;
            }
        }
    }

    if (ix1 < 0)
    {
        ix0 = (int)ox; ix1 = (int)(ox + Math.Max(1, advance)); iy0 = (int)oy; iy1 = (int)(oy + em);
    }

    ix1++; iy1++;
    float inkH = iy1 - iy0;
    float hh = MathF.Max(inkH, 0.85f * (float)p.Ref * refH);
    float cy = (iy0 + iy1) / 2f;
    int x0 = Math.Max(0, (int)MathF.Floor(ix0 - (float)p.Hm[0] * hh));
    int x1 = Math.Min(w, (int)MathF.Ceiling(ix1 + (float)p.Hm[1] * hh));
    int y0 = Math.Max(0, (int)MathF.Floor(cy - hh / 2 - (float)p.Vm[0] * hh));
    int y1 = Math.Min(h, (int)MathF.Ceiling(cy + hh / 2 + (float)p.Vm[1] * hh));
    Image crop = gray.Crop(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));

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

static float RefInkHeight(Font font)
{
    var o = new TextOptions(font);
    float top = float.MaxValue, bottom = float.MinValue;
    foreach (var g in TextMeasurer.GetCharacterBounds("dlgpy", o))
    {
        if (g.Ink.Height <= 0) continue;
        top = MathF.Min(top, g.Ink.Top);
        bottom = MathF.Max(bottom, g.Ink.Bottom);
    }

    return bottom > top ? bottom - top : font.Size;
}

static int RunPageLines(string path, string outDir, int height, ParallelOptions po)
{
    foreach (var sub in new[] { "finder", "box", "margin" })
    {
        Directory.CreateDirectory(Path.Combine(outDir, sub, "images"));
    }

    var lines = File.ReadLines(path).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<PageLine>(l, PlanJson.Options)!).ToArray();
    var pages = new Dictionary<string, (Image Img, Gray8[] Px, int W, int H)>();
    foreach (string pg in lines.Select(l => l.Page).Distinct())
    {
        Image im = Image.Load(pg).ConvertTo(PixelFormat.Gray8);
        pages[pg] = (im, im.ToArray<Gray8>(), im.Width, im.Height);
    }

    // Ink box of each line inside its truth box: pixels darker than the mid level between the darkest and lightest pixel. Truth
    // boxes are layout boxes and can reach into the neighbouring lines, so rows are chosen by horizontal projection (as the
    // library's line finder does): the run of rows with ink, gaps of up to 2 rows bridged, that holds the densest row.
    var ink = new (int X0, int Y0, int X1, int Y1)[lines.Length];
    Parallel.For(0, lines.Length, po, i =>
    {
        var l = lines[i];
        var (_, px, w, h) = pages[l.Page];
        int bx0 = Math.Clamp((int)Math.Floor(l.Box[0]), 0, w - 1), bx1 = Math.Clamp((int)Math.Ceiling(l.Box[2]), 1, w);
        int by0 = Math.Clamp((int)Math.Floor(l.Box[1]), 0, h - 1), by1 = Math.Clamp((int)Math.Ceiling(l.Box[3]), 1, h);
        int lo = 255, hi = 0;
        for (int y = by0; y < by1; y++)
            for (int x = bx0; x < bx1; x++) { int v = px[y * w + x].V; lo = Math.Min(lo, v); hi = Math.Max(hi, v); }
        int thr = (lo + hi) / 2;
        var rows = new int[by1 - by0];
        for (int y = by0; y < by1; y++)
            for (int x = bx0; x < bx1; x++)
                if (px[y * w + x].V < thr) rows[y - by0]++;
        int peak = 0;
        for (int y = 1; y < rows.Length; y++) if (rows[y] > rows[peak]) peak = y;
        if (rows.Length == 0 || rows[peak] == 0) { ink[i] = (bx0, by0, bx1, by1); return; }
        int minRow = Math.Max(1, (int)(0.04 * rows[peak]));
        int top = peak, bot = peak;
        for (int y = peak - 1, gap = 0; y >= 0 && gap <= 2; y--) { if (rows[y] >= minRow) { top = y; gap = 0; } else gap++; }
        for (int y = peak + 1, gap = 0; y < rows.Length && gap <= 2; y++) { if (rows[y] >= minRow) { bot = y; gap = 0; } else gap++; }
        int x0 = int.MaxValue, x1 = -1;
        for (int y = by0 + top; y <= by0 + bot; y++)
            for (int x = bx0; x < bx1; x++)
                if (px[y * w + x].V < thr) { x0 = Math.Min(x0, x); x1 = Math.Max(x1, x); }
        ink[i] = x1 < 0 ? (bx0, by0, bx1, by1) : (x0, by0 + top, x1 + 1, by0 + bot + 1);
    });

    var medianH = lines.Select((l, i) => (l.Page, H: ink[i].Y1 - ink[i].Y0)).GroupBy(t => t.Page)
        .ToDictionary(g => g.Key, g => { var s = g.Select(t => t.H).OrderBy(v => v).ToArray(); return s[s.Length / 2]; });

    var finder = new string[lines.Length];
    var box = new string[lines.Length];
    var margin = new string[lines.Length];
    Parallel.For(0, lines.Length, po, i =>
    {
        var l = lines[i];
        var (img, _, w, h) = pages[l.Page];
        var b = ink[i];
        double hh = Math.Max(b.Y1 - b.Y0, 0.85 * medianH[l.Page]);
        double cy = (b.Y0 + b.Y1) / 2.0;
        int x0 = Math.Max(0, (int)Math.Floor(b.X0 - hh / 2)), x1 = Math.Min(w, (int)Math.Ceiling(b.X1 + hh / 2));
        int y0 = Math.Max(0, (int)Math.Floor(cy - hh * 0.75)), y1 = Math.Min(h, (int)Math.Ceiling(cy + hh * 0.75));
        finder[i] = Save(img.Crop(x0, y0, x1 - x0, y1 - y0).Resize(height: height), Path.Combine(outDir, "finder"), "images", l.Id, l.Text, "page");
        int tx0 = Math.Clamp((int)Math.Floor(l.Box[0]), 0, w - 1), tx1 = Math.Clamp((int)Math.Ceiling(l.Box[2]), tx0 + 1, w);
        int ty0 = Math.Clamp((int)Math.Floor(l.Box[1]), 0, h - 1), ty1 = Math.Clamp((int)Math.Ceiling(l.Box[3]), ty0 + 1, h);
        box[i] = Save(img.Crop(tx0, ty0, tx1 - tx0, ty1 - ty0).Resize(height: height), Path.Combine(outDir, "box"), "images", l.Id, l.Text, "page");
        // the truth box with the library's crop margins (H/2 left and right, H/4 above and below), for pages where ink finding
        // would also pick up text the truth does not cover
        double bh = ty1 - ty0;
        int mx0 = Math.Max(0, (int)Math.Floor(tx0 - bh / 2)), mx1 = Math.Min(w, (int)Math.Ceiling(tx1 + bh / 2));
        int my0 = Math.Max(0, (int)Math.Floor(ty0 - bh / 4)), my1 = Math.Min(h, (int)Math.Ceiling(ty1 + bh / 4));
        margin[i] = Save(img.Crop(mx0, my0, mx1 - mx0, my1 - my0).Resize(height: height), Path.Combine(outDir, "margin"), "images", l.Id, l.Text, "page");
    });
    File.WriteAllLines(Path.Combine(outDir, "finder", "labels.jsonl"), finder, new UTF8Encoding(false));
    File.WriteAllLines(Path.Combine(outDir, "box", "labels.jsonl"), box, new UTF8Encoding(false));
    File.WriteAllLines(Path.Combine(outDir, "margin", "labels.jsonl"), margin, new UTF8Encoding(false));
    Console.WriteLine($"{lines.Length} page lines -> {outDir}");
    return 0;
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

record PlanLine(string Id, string Text, string Font, double Em, int Fg, int Bg, double Ref, double[] Vm, double[] Hm, double Blur, double Noise, int Jpeg, long Seed);

record PageLine(string Id, string Text, string Page, double[] Box);

record PackLine(string Id, string Text, int W, int H, long Off, string Font, string Sha256);

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
