using System.Globalization;

namespace OneOcr.Model;

/// <summary>The scripts a recognizer calibrates separately.</summary>
public enum OcrScript
{
    /// <summary>Latin letters, and every letter that is neither Greek nor Cyrillic.</summary>
    Latin = 0,

    /// <summary>Greek and Coptic, and Greek Extended.</summary>
    Greek = 1,

    /// <summary>Cyrillic and its supplements and extensions.</summary>
    Cyrillic = 2,
}

internal sealed class CalibrationDto
{
    public required string Raw { get; init; }
    public required Dictionary<string, KnotsDto> Scripts { get; init; }
}

internal sealed class KnotsDto
{
    public required List<double> X { get; init; }
    public required List<double> Y { get; init; }
}

/// <summary>
/// Maps a word's raw confidence to a calibrated one: one monotone piecewise-linear curve per script, read from the
/// <c>calibration</c> member of a version 1 model container (docs/spec/model-container-v1.md). Immutable and thread-safe.
/// </summary>
public sealed class WordCalibration
{
    /// <summary>The only raw confidence definition containers may name.</summary>
    internal const string RawDefinition = "word-mean-max-run";

    private readonly double[]?[] _x = new double[]?[3];
    private readonly double[]?[] _y = new double[]?[3];

    private WordCalibration()
    {
    }

    /// <summary>Whether the model carries a curve for this script. A script without one maps raw confidence to itself.</summary>
    /// <param name="script">The script.</param>
    /// <returns><see langword="true"/> when a curve is present.</returns>
    public bool HasCurve(OcrScript script) => (uint)script < 3 && _x[(int)script] is not null;

    /// <summary>Calibrates a raw word confidence.</summary>
    /// <param name="script">The word's script.</param>
    /// <param name="raw">The raw confidence in 0..1.</param>
    /// <returns>
    /// The curve's value at <paramref name="raw"/>: linear between the two knots around it, the first or last knot's y outside
    /// the knot range. Computed in double precision with separate multiply and add, then rounded to float, so every
    /// platform agrees. Without a curve for <paramref name="script"/>, <paramref name="raw"/> itself.
    /// </returns>
    public float Apply(OcrScript script, float raw)
    {
        if (!HasCurve(script))
        {
            return raw;
        }

        double[] xs = _x[(int)script]!, ys = _y[(int)script]!;
        double x = raw;
        if (!(x > xs[0]))
        {
            return (float)ys[0];
        }

        int last = xs.Length - 1;
        if (x >= xs[last])
        {
            return (float)ys[last];
        }

        int i = 0;
        while (xs[i + 1] <= x)
        {
            i++;
        }

        double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
        double product = (ys[i + 1] - ys[i]) * t;
        return (float)(ys[i] + product);
    }

    /// <summary>Validates the parsed member and builds the curves; throws for anything the spec forbids.</summary>
    internal static WordCalibration FromDto(CalibrationDto dto, OcrLimits limits)
    {
        if (dto.Raw != RawDefinition)
        {
            throw new OcrModelException($"Calibration names the raw confidence '{dto.Raw}'; only '{RawDefinition}' is defined.");
        }

        var result = new WordCalibration();
        foreach ((string name, KnotsDto knots) in dto.Scripts)
        {
            OcrScript script;
            switch (name)
            {
                case "Latin":
                    script = OcrScript.Latin;
                    break;
                case "Greek":
                    script = OcrScript.Greek;
                    break;
                case "Cyrillic":
                    script = OcrScript.Cyrillic;
                    break;
                default:
                    throw new OcrModelException($"Calibration names an unknown script '{name}'; the scripts are Latin, Greek and Cyrillic.");
            }

            if (knots is null || knots.X is null || knots.Y is null)
            {
                throw new OcrModelException($"Calibration for {name} needs 'x' and 'y' arrays.");
            }

            if (knots.X.Count != knots.Y.Count)
            {
                throw new OcrModelException($"Calibration for {name}: x has {knots.X.Count} knots but y has {knots.Y.Count}.");
            }

            if (knots.X.Count == 0)
            {
                throw new OcrModelException($"Calibration for {name} has no knots.");
            }

            if (knots.X.Count > limits.MaxCalibrationKnots)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxCalibrationKnots), limits.MaxCalibrationKnots, knots.X.Count);
            }

            for (int i = 0; i < knots.X.Count; i++)
            {
                double x = knots.X[i], y = knots.Y[i];
                if (!(x >= 0 && x <= 1) || !(y >= 0 && y <= 1))
                {
                    throw new OcrModelException(string.Create(CultureInfo.InvariantCulture, $"Calibration for {name}: knot {i} ({x}, {y}) is outside [0, 1]."));
                }

                if (i > 0 && !(x > knots.X[i - 1]))
                {
                    throw new OcrModelException($"Calibration for {name}: x must be strictly increasing (knot {i}).");
                }

                if (i > 0 && y < knots.Y[i - 1])
                {
                    throw new OcrModelException($"Calibration for {name}: y must not decrease (knot {i}).");
                }
            }

            result._x[(int)script] = [.. knots.X];
            result._y[(int)script] = [.. knots.Y];
        }

        return result;
    }
}

/// <summary>Assigns a word to a script by counting its letters.</summary>
internal static class ScriptDetector
{
    /// <summary>
    /// The script with the most letters in <paramref name="word"/>; ties go to the first of Latin, Greek, Cyrillic. A word
    /// with no letters is Latin. Greek is U+0370–03FF and U+1F00–1FFF; Cyrillic is U+0400–052F, U+1C80–1C8F, U+2DE0–2DFF and
    /// U+A640–A69F; every other letter counts as Latin.
    /// </summary>
    public static OcrScript Of(string word)
    {
        int latin = 0, greek = 0, cyrillic = 0;
        foreach (System.Text.Rune r in word.EnumerateRunes())
        {
            if (!System.Text.Rune.IsLetter(r))
            {
                continue;
            }

            int v = r.Value;
            if ((v >= 0x370 && v <= 0x3FF) || (v >= 0x1F00 && v <= 0x1FFF))
            {
                greek++;
            }
            else if ((v >= 0x400 && v <= 0x52F) || (v >= 0x1C80 && v <= 0x1C8F) || (v >= 0x2DE0 && v <= 0x2DFF) || (v >= 0xA640 && v <= 0xA69F))
            {
                cyrillic++;
            }
            else
            {
                latin++;
            }
        }

        if (latin >= greek && latin >= cyrillic)
        {
            return OcrScript.Latin;
        }

        return greek >= cyrillic ? OcrScript.Greek : OcrScript.Cyrillic;
    }
}
