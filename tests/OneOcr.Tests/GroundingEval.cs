using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using OneOcr.Llm;

namespace OneOcr.Tests;

/// <summary>One field with exact truth: what is printed, which words hold it, and how a model would write it.</summary>
public sealed record EvalField(string Path, string Printed, int[] WordIdx, bool IsNumber, decimal? Amount, string? IsoDate);

public sealed record EvalDoc(string Id, OcrPage Page, IReadOnlyList<EvalField> Fields);

public enum Variant { Correct, Case, WrongCite, NoCite, Halluc, DerivedVerified, DerivedGuess, Null }

public sealed record Planned(EvalField Field, Variant Variant, string? ValueJson, string[] Cited, bool Derived, GroundingState? Expect, bool ExpectCorrected);

/// <summary>Generated pages with exact truth, scripted model answers and the scoring of grounding against them.</summary>
public static class GroundingEval
{
    private static readonly string[] Vendors = ["Acme", "Northwind", "Contoso", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Wonka", "Soylent", "Tyrell"];
    private static readonly string[] Suffix = ["Ltd", "GmbH", "Inc", "SARL", "BV", "Supplies", "Trading"];
    private static readonly string[] Goods = ["Widget", "Gadget", "Bracket", "Cable", "Sensor", "Filter", "Adapter", "Bolt", "Panel", "Valve", "Lamp", "Tape"];
    private static readonly string[] Months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

    // ---- generated forms and receipts ----

    public static List<EvalDoc> Generate(int count, int seed)
    {
        var docs = new List<EvalDoc>();
        for (int d = 0; d < count; d++)
        {
            var rng = new Random(seed + d);
            docs.Add(GenerateOne(rng, d));
        }

        return docs;
    }

    private static string Amount(decimal v, int style)
    {
        string plain = v.ToString("0.00", CultureInfo.InvariantCulture);
        if (v < 1000)
        {
            return style == 0 ? plain : plain.Replace('.', ',');
        }

        string whole = ((long)v).ToString(CultureInfo.InvariantCulture);
        string frac = plain[^2..];
        string grouped = whole.Length > 3 ? whole[..^3] + (style switch { 0 => ",", 1 => ".", _ => " " }) + whole[^3..] : whole;
        return grouped + (style == 0 ? "." : ",") + frac;
    }

    private static EvalDoc GenerateOne(Random rng, int n)
    {
        int style = rng.Next(3);
        bool invoice = rng.Next(2) == 0;
        string cur = new[] { "EUR", "USD", "GBP", "CHF" }[rng.Next(4)];
        var lines = new List<List<(string Token, int Field)>>();
        var fields = new List<(string Printed, bool IsNumber, decimal? Amount, string? Iso)>();
        int Tag(string printed, bool number = false, decimal? amount = null, string? iso = null)
        {
            fields.Add((printed, number, amount, iso));
            return fields.Count - 1;
        }

        (string, int) T(string s) => (s, -1);
        IEnumerable<(string, int)> Tokens(string printed, int id) => printed.Split(' ').Select(t => (t, id));

        string vendor = Vendors[rng.Next(Vendors.Length)] + " " + Suffix[rng.Next(Suffix.Length)];
        lines.Add([.. Tokens(vendor, Tag(vendor))]);
        DateTime date = new DateTime(2024, 1, 1).AddDays(rng.Next(0, 900));
        string iso = date.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        string printedDate = rng.Next(5) switch
        {
            0 => iso,
            1 => date.ToString("dd.MM.yyyy", CultureInfo.InvariantCulture),
            2 => $"{date.Day} {Months[date.Month - 1]} {date.Year}",
            3 => $"{Months[date.Month - 1][..3]} {date.Day}, {date.Year}",
            _ => date.ToString("dd/MM/yyyy", CultureInfo.InvariantCulture),
        };
        if (invoice)
        {
            string number = "INV-" + rng.Next(1000, 9999).ToString(CultureInfo.InvariantCulture);
            lines.Add([T("Invoice"), T("No:"), (number, Tag(number))]);
        }

        lines.Add([T(invoice ? "Date:" : "Receipt"), .. Tokens(printedDate, Tag(printedDate, iso: iso))]);
        decimal total = 0;
        int items = rng.Next(2, 5);
        for (int i = 0; i < items; i++)
        {
            string desc = Goods[rng.Next(Goods.Length)] + (rng.Next(2) == 0 ? " " + (char)('A' + rng.Next(6)) : "");
            decimal price = Math.Round((decimal)(rng.NextDouble() * (rng.Next(3) == 0 ? 2500 : 90)) + 1, 2);
            total += price;
            lines.Add([.. Tokens(desc, Tag(desc)), T(rng.Next(1, 4).ToString(CultureInfo.InvariantCulture) + "x"), (Amount(price, style), Tag(Amount(price, style), true, price))]);
        }

        decimal tax = Math.Round(total * 0.19m, 2);
        string st = Amount(total, style), tx = Amount(tax, style), tt = Amount(total + tax, style);
        lines.Add([T("Subtotal"), (st, Tag(st, true, total))]);
        lines.Add([T("Tax"), T("19%"), (tx, Tag(tx, true, tax))]);
        lines.Add([T("Total"), (cur, Tag(cur)), (tt, Tag(tt, true, total + tax))]);

        OcrPage page = GroundingSupport.Layout([.. lines.Select(l => new TestLine(string.Join(' ', l.Select(static t => t.Token))))]);
        var idx = new Dictionary<int, List<int>>();
        int wi = 0;
        foreach (var l in lines)
        {
            foreach ((string _, int f) in l)
            {
                if (f >= 0)
                {
                    if (!idx.TryGetValue(f, out List<int>? list))
                    {
                        idx[f] = list = [];
                    }

                    list.Add(wi);
                }

                wi++;
            }
        }

        var ef = new List<EvalField>();
        for (int f = 0; f < fields.Count; f++)
        {
            ef.Add(new EvalField("/f" + f, fields[f].Printed, [.. idx[f]], fields[f].IsNumber, fields[f].Amount, fields[f].Iso));
        }

        return new EvalDoc("gen" + n.ToString(CultureInfo.InvariantCulture), page, ef);
    }

    // ---- scripted answers ----

    private static string Digits(string s) => new(s.Where(char.IsAsciiDigit).ToArray());

    private static bool DigitsOnPage(OcrPage page, string value)
    {
        string d = Digits(value);
        for (int i = 0; i < page.Words.Count; i++)
        {
            string acc = "";
            for (int k = i; k < Math.Min(page.Words.Count, i + 3); k++)
            {
                acc += Digits(page.Words[k].Text);
                if (acc.Length > 2 && acc == d)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool OnPage(OcrPage page, string value)
    {
        string v = value.Replace(" ", "", StringComparison.Ordinal).ToLowerInvariant();
        IReadOnlyList<OcrWord> w = page.Words;
        for (int i = 0; i < w.Count; i++)
        {
            string acc = "";
            for (int k = i; k < Math.Min(w.Count, i + 8); k++)
            {
                acc += w[k].Text.Replace(" ", "", StringComparison.Ordinal).ToLowerInvariant();
                if (acc == v || acc.Trim('.', ',', ':') == v)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static string Mutate(Random rng, string s)
    {
        for (int tries = 0; tries < 20; tries++)
        {
            var chars = s.ToCharArray();
            int i = rng.Next(chars.Length);
            chars[i] = char.IsAsciiDigit(chars[i]) ? (char)('0' + ((chars[i] - '0' + rng.Next(1, 10)) % 10)) : char.IsAsciiLetter(chars[i]) ? (char)((char.IsUpper(chars[i]) ? 'A' : 'a') + ((char.ToLowerInvariant(chars[i]) - 'a' + rng.Next(1, 26)) % 26)) : '#';
            string m = new(chars);
            if (m != s)
            {
                return m;
            }
        }

        return s + "7";
    }

    public static List<Planned> Plan(EvalDoc doc, Random rng)
    {
        var plan = new List<Planned>();
        foreach (EvalField f in doc.Fields)
        {
            int roll = rng.Next(100);
            string[] truthIds = [.. f.WordIdx.Select(static i => "w" + i)];
            string printedNoWs = f.Printed.Replace(" ", "", StringComparison.Ordinal);
            // The value a model writes for this field when it is right.
            string rightJson;
            string rightText;
            if (f.IsNumber && f.Amount is { } amt)
            {
                rightText = amt.ToString(CultureInfo.InvariantCulture);
                rightJson = rightText;
            }
            else if (f.IsoDate is { } iso)
            {
                rightText = iso;
                rightJson = GroundingSupport.S(iso);
            }
            else
            {
                rightText = f.Printed;
                rightJson = GroundingSupport.S(f.Printed);
            }

            bool contiguous = f.WordIdx.Length < 2 || OnPage(doc.Page, f.Printed);
            GroundingState exact = contiguous && rightText.Replace(" ", "", StringComparison.Ordinal) == printedNoWs ? GroundingState.Grounded : GroundingState.GroundedNormalized;
            string[] stray = [.. Enumerable.Range(0, 3).Select(_ => "w" + rng.Next(doc.Page.Words.Count)).Where(id => !truthIds.Contains(id) && !f.WordIdx.Contains(int.Parse(id[1..], CultureInfo.InvariantCulture))).Distinct()];
            if (roll < 30)
            {
                plan.Add(new Planned(f, Variant.Correct, rightJson, truthIds, false, exact, false));
            }
            else if (roll < 40 && !f.IsNumber && f.IsoDate is null && f.Printed.Any(char.IsAsciiLetter) && f.Printed.ToUpperInvariant() != f.Printed)
            {
                plan.Add(new Planned(f, Variant.Case, GroundingSupport.S(f.Printed.ToUpperInvariant()), truthIds, false, GroundingState.GroundedNormalized, false));
            }
            else if (roll < 55)
            {
                plan.Add(new Planned(f, Variant.WrongCite, rightJson, stray, false, exact, true));
            }
            else if (roll < 65)
            {
                plan.Add(new Planned(f, Variant.NoCite, rightJson, [], false, exact, true));
            }
            else if (roll < 80 && f.Printed.Length > 2)
            {
                string bad;
                string badJson;
                int tries = 0;
                do
                {
                    if (f.IsNumber && f.Amount is { } a)
                    {
                        decimal v = Math.Round(a + (decimal)(rng.Next(1, 9999) / 100.0), 2);
                        bad = v.ToString(CultureInfo.InvariantCulture);
                        badJson = bad;
                    }
                    else
                    {
                        bad = Mutate(rng, f.IsoDate ?? f.Printed);
                        badJson = GroundingSupport.S(bad);
                    }
                }
                while (++tries < 10 && (OnPage(doc.Page, bad) || (f.IsNumber && DigitsOnPage(doc.Page, decimal.Parse(bad, CultureInfo.InvariantCulture).ToString("0.00", CultureInfo.InvariantCulture)))));
                plan.Add(new Planned(f, Variant.Halluc, badJson, stray.Length > 0 ? stray : truthIds, false, GroundingState.Ungrounded, false));
            }
            else if (roll < 85 && f.IsNumber && doc.Fields.Count(g => g.IsNumber) >= 2)
            {
                EvalField[] others = [.. doc.Fields.Where(g => g.IsNumber && g.Path != f.Path).Take(2)];
                decimal sum = others.Sum(static g => g.Amount ?? 0m);
                string text = sum.ToString(CultureInfo.InvariantCulture);
                bool printed = DigitsOnPage(doc.Page, sum.ToString("0.00", CultureInfo.InvariantCulture));
                plan.Add(printed
                    ? new Planned(f, Variant.Correct, rightJson, truthIds, false, exact, false)
                    : new Planned(f, Variant.DerivedVerified, text, [.. others.SelectMany(static g => g.WordIdx).Select(static i => "w" + i)], true, GroundingState.Derived, false));
            }
            else if (roll < 92)
            {
                string bad = f.IsNumber ? (rng.Next(100000, 999999) / 100m).ToString(CultureInfo.InvariantCulture) : "Zq" + rng.Next(1000, 9999).ToString(CultureInfo.InvariantCulture) + " Xv";
                plan.Add(new Planned(f, Variant.DerivedGuess, f.IsNumber ? bad : GroundingSupport.S(bad), truthIds, true, GroundingState.Derived, false));
            }
            else
            {
                plan.Add(new Planned(f, Variant.Null, "null", [], false, null, false));
            }
        }

        // DerivedVerified stores plain numbers: fix JSON text for strings.
        return plan;
    }

    public static (string Schema, string Answer) Build(IReadOnlyList<Planned> plan)
    {
        var props = new StringBuilder();
        var data = new StringBuilder();
        foreach (Planned p in plan)
        {
            string name = p.Field.Path[1..];
            string type = p.Field.IsNumber ? "number" : "string";
            if (props.Length > 0)
            {
                props.Append(',');
                data.Append(',');
            }

            props.Append('"').Append(name).Append("\":{\"type\":[\"").Append(type).Append("\",\"null\"]}");
            data.Append('"').Append(name).Append("\":").Append(GroundingSupport.V(p.ValueJson ?? "null", p.Cited, p.Derived));
        }

        return ("{\"type\":\"object\",\"properties\":{" + props + "}}", "{\"data\":{" + data + "}}");
    }

    // ---- scoring ----

    public sealed class Tally
    {
        public int Total, StateOk, BoxChecked, BoxOk, CorrectedChecked, CorrectedOk;
        public Dictionary<Variant, (int N, int Ok)> ByVariant = [];
        public int HallTp, HallFp, HallFn;
        public Dictionary<GroundingState, (int N, double Conf)> Conf = [];
        public int Docs, ReaskCalls;
        public List<string> Misses = [];

        public string Report(string title)
        {
            double P(int a, int b) => b == 0 ? double.NaN : (double)a / b;
            var sb = new StringBuilder();
            sb.Append("### ").AppendLine(title);
            sb.AppendLine($"documents {Docs}, fields {Total}, re-ask calls {ReaskCalls}");
            sb.AppendLine($"state accuracy {P(StateOk, Total):P2} ({StateOk}/{Total})");
            sb.AppendLine($"ungrounded detection: precision {P(HallTp, HallTp + HallFp):P2}, recall {P(HallTp, HallTp + HallFn):P2} (tp {HallTp}, fp {HallFp}, fn {HallFn})");
            sb.AppendLine($"box on the right place (IoU >= 0.5 with a true occurrence): {P(BoxOk, BoxChecked):P2} ({BoxOk}/{BoxChecked})");
            sb.AppendLine($"citation correction flagged where the citation was wrong or missing: {P(CorrectedOk, CorrectedChecked):P2} ({CorrectedOk}/{CorrectedChecked})");
            sb.AppendLine("| variant | fields | state correct |");
            sb.AppendLine("|---|---|---|");
            foreach ((Variant v, (int n, int ok)) in ByVariant.OrderBy(static kv => kv.Key))
            {
                sb.AppendLine($"| {v} | {n} | {P(ok, n):P1} |");
            }

            sb.AppendLine("| state | fields | mean confidence |");
            sb.AppendLine("|---|---|---|");
            foreach ((GroundingState s, (int n, double c)) in Conf.OrderBy(static kv => kv.Key))
            {
                sb.AppendLine($"| {s} | {n} | {c / n:F2} |");
            }

            foreach (string m in Misses)
            {
                sb.Append("miss: ").AppendLine(m);
            }

            return sb.ToString();
        }
    }

    private static double Iou(OcrBox a, OcrBox b)
    {
        double w = Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left), h = Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top);
        if (w <= 0 || h <= 0)
        {
            return 0;
        }

        double inter = w * h;
        return inter / ((a.Width * a.Height) + (b.Width * b.Height) - inter);
    }

    private static OcrBox Union(IEnumerable<OcrBox> boxes) =>
        new(boxes.Min(static b => b.Left), boxes.Min(static b => b.Top), boxes.Max(static b => b.Right), boxes.Max(static b => b.Bottom));

    public static async Task<Tally> RunAsync(IEnumerable<EvalDoc> docs, int seed, string provider = "openai")
    {
        var tally = new Tally();
        int di = 0;
        foreach (EvalDoc doc in docs)
        {
            var rng = new Random(seed + (di++ * 7919));
            List<Planned> plan = Plan(doc, rng);
            (string schema, string answer) = Build(plan);
            var h = new ScriptedHandler().Then(GroundingSupport.Shape(provider, answer));
            using var http = new HttpClient(h);
            using var client = new LlmClient(GroundingSupport.Provider(provider, http), GroundingSupport.Options());
            LlmReadResult r = await client.ReadAsync(GroundingSupport.Blank(doc.Page), doc.Page, new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = schema });
            tally.Docs++;
            tally.ReaskCalls += r.Calls - 1;
            foreach (Planned p in plan)
            {
                GroundedField? got = r.Fields.FirstOrDefault(f => f.Path == p.Field.Path);
                GroundingState? actual = got?.State;
                tally.Total++;
                bool ok = actual == p.Expect;
                if (ok)
                {
                    tally.StateOk++;
                }
                else if (tally.Misses.Count < 25)
                {
                    tally.Misses.Add($"{doc.Id}{p.Field.Path} {p.Variant} value={p.ValueJson} printed='{p.Field.Printed}' expected={p.Expect} got={actual} note={got?.Note}");
                }

                (int n, int okv) = tally.ByVariant.GetValueOrDefault(p.Variant);
                tally.ByVariant[p.Variant] = (n + 1, okv + (ok ? 1 : 0));
                bool truthBad = p.Expect == GroundingState.Ungrounded;
                bool predBad = actual == GroundingState.Ungrounded;
                if (truthBad && predBad)
                {
                    tally.HallTp++;
                }
                else if (!truthBad && predBad)
                {
                    tally.HallFp++;
                }
                else if (truthBad && !predBad)
                {
                    tally.HallFn++;
                }

                if (got is not null)
                {
                    (int cn, double cc) = tally.Conf.GetValueOrDefault(got.State);
                    tally.Conf[got.State] = (cn + 1, cc + got.Confidence);
                }

                if (got is { State: GroundingState.Grounded or GroundingState.GroundedNormalized } && p.Expect is GroundingState.Grounded or GroundingState.GroundedNormalized)
                {
                    tally.BoxChecked++;
                    // Any place on the page that holds the same text counts as right.
                    OcrBox truth = Union(p.Field.WordIdx.Select(i => doc.Page.Words[i].Box));
                    OcrBox mine = Union(got.Boxes.Select(static b => b.Box));
                    bool same = Iou(truth, mine) >= 0.5 || SameTextElsewhere(doc.Page, p.Field, mine);
                    if (same)
                    {
                        tally.BoxOk++;
                    }

                    if (p.Variant is Variant.WrongCite or Variant.NoCite)
                    {
                        tally.CorrectedChecked++;
                        if (got.CitationCorrected)
                        {
                            tally.CorrectedOk++;
                        }
                    }
                }
            }
        }

        return tally;
    }

    private static bool SameTextElsewhere(OcrPage page, EvalField f, OcrBox mine)
    {
        string printed = f.Printed.Replace(" ", "", StringComparison.Ordinal);
        for (int i = 0; i < page.Words.Count; i++)
        {
            string acc = "";
            for (int k = i; k < Math.Min(page.Words.Count, i + 8); k++)
            {
                acc += page.Words[k].Text.Replace(" ", "", StringComparison.Ordinal);
                if (acc == printed && Iou(Union(Enumerable.Range(i, k - i + 1).Select(j => page.Words[j].Box)), mine) >= 0.5)
                {
                    return true;
                }
            }
        }

        return false;
    }
}

public class GroundingEvalTests
{
    [Fact]
    public async Task Grounding_labels_scripted_answers_correctly_on_generated_forms_and_receipts_for_every_provider_shape()
    {
        var report = new StringBuilder();
        foreach (string provider in new[] { "openai", "anthropic" })
        {
            GroundingEval.Tally t = await GroundingEval.RunAsync(GroundingEval.Generate(300, 1000), 5, provider);
            report.AppendLine(t.Report($"generated forms and receipts, scripted answers, {provider} reply shape"));
            Assert.True((double)t.StateOk / t.Total >= 0.99, report.ToString());
            Assert.True((double)t.HallTp / (t.HallTp + t.HallFn) >= 0.99, report.ToString());
            Assert.True((double)t.HallTp / (t.HallTp + t.HallFp) >= 0.99, report.ToString());
            Assert.True((double)t.BoxOk / t.BoxChecked >= 0.99, report.ToString());
        }

        Environment.GetEnvironmentVariable("ONEOCR_GROUNDING_EVAL_OUT").AppendReport(report.ToString());
    }

    [Fact]
    public async Task Grounding_labels_scripted_answers_correctly_on_cord_test_receipts()
    {
        string? path = Environment.GetEnvironmentVariable("ONEOCR_CORD_EVAL");
        Assert.SkipWhen(path is null || !File.Exists(path), "ONEOCR_CORD_EVAL (CORD v2 test words and fields as JSON) is not set.");
        List<EvalDoc> docs = CordEval.Load(path!);
        GroundingEval.Tally t = await GroundingEval.RunAsync(docs, 11);
        string report = t.Report("CORD v2 test receipts (CC BY 4.0), scripted answers");
        Environment.GetEnvironmentVariable("ONEOCR_GROUNDING_EVAL_OUT").AppendReport(report);
        Assert.True((double)t.StateOk / t.Total >= 0.97, report);
        Assert.True((double)t.HallTp / (t.HallTp + t.HallFn) >= 0.97, report);
    }
}

internal static class ReportExtensions
{
    public static void AppendReport(this string? path, string text)
    {
        if (!string.IsNullOrEmpty(path))
        {
            File.AppendAllText(path, text + "\n");
        }
    }
}

/// <summary>Reads CORD v2 test receipts converted by <c>tests/grounding-eval/cord_to_json.py</c>: printed words with boxes and the annotated fields.</summary>
public static class CordEval
{
    public static List<EvalDoc> Load(string path)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        var docs = new List<EvalDoc>();
        foreach (JsonElement r in doc.RootElement.EnumerateArray())
        {
            var words = new List<OcrWord>();
            foreach (JsonElement w in r.GetProperty("words").EnumerateArray())
            {
                JsonElement b = w.GetProperty("box");
                words.Add(new OcrWord(w.GetProperty("text").GetString()!, new OcrBox(b[0].GetDouble(), b[1].GetDouble(), b[2].GetDouble(), b[3].GetDouble()), 0.95f, 0.95f, OcrSource.Ocr, []));
            }

            if (words.Count == 0)
            {
                continue;
            }

            // One line per word run keeps the page model valid; grounding looks at the words in reading order only.
            var line = new OcrLine(string.Join(' ', words.Select(static w => w.Text)), words, [0], new OcrBox(words.Min(static w => w.Box.Left), words.Min(static w => w.Box.Top), words.Max(static w => w.Box.Right), words.Max(static w => w.Box.Bottom)), 0.95f, OcrSource.Ocr);
            double width = r.GetProperty("width").GetDouble(), height = r.GetProperty("height").GetDouble();
            var page = new OcrPage(0, width, height, 96, [new OcrBlock(OcrBlockRole.Body, false, line.Box, [line], 0.95f)]);
            var fields = new List<EvalField>();
            foreach (JsonElement f in r.GetProperty("fields").EnumerateArray())
            {
                int[] idx = [.. f.GetProperty("words").EnumerateArray().Select(static e => e.GetInt32())];
                fields.Add(new EvalField("/f" + fields.Count.ToString(CultureInfo.InvariantCulture), string.Join(' ', idx.Select(i => words[i].Text)), idx, false, null, null));
            }

            if (fields.Count > 0)
            {
                docs.Add(new EvalDoc(r.GetProperty("id").GetString()!, page, fields));
            }
        }

        return docs;
    }
}
