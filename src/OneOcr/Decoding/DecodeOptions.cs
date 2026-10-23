namespace OneOcr.Decoding;

/// <summary>How the recognizer's per-step class probabilities become text.</summary>
public enum CtcDecoding
{
    /// <summary>Arg-max per step, merge repeats, drop blanks. The default; no language model or lexicon applies.</summary>
    Greedy = 0,

    /// <summary>
    /// Prefix beam search over all alignments (Graves et al. 2006; Hannun et al. 2014), optionally ranked with a character
    /// language model and a lexicon.
    /// </summary>
    Beam = 1,
}

/// <summary>
/// Decoding settings, immutable once built. The default is <see cref="CtcDecoding.Greedy"/>: the fastest mode and the one
/// every other mode is compared against. <see cref="CtcDecoding.Beam"/> without a model or lexicon sums over alignments, which
/// rarely changes text. With a <see cref="LanguageModel"/> it prefers readings that look like words of the script.
/// Words that contain a digit or a code symbol (<c>@ # _ / \ = % + * ~ ^ | &lt; &gt; [ ] { }</c>) or a character the model does not
/// know are never scored by the language model, so numbers, identifiers, amounts and dates are read as the recognizer saw them.
/// </summary>
public sealed class DecodeOptions
{
    private volatile Prepared? _prepared;

    /// <summary>Greedy decoding.</summary>
    public static DecodeOptions Greedy { get; } = new();

    /// <summary>Greedy or beam search. Default <see cref="CtcDecoding.Greedy"/>; everything below applies only to beam search.</summary>
    public CtcDecoding Mode { get; init; } = CtcDecoding.Greedy;

    /// <summary>Prefixes kept after each step, 1 to <see cref="OcrLimits.MaxBeamWidth"/>. Default 8.</summary>
    public int BeamWidth { get; init; } = 8;

    /// <summary>The character model, or null for none. The script whose alphabet covers most letters of the line is used.</summary>
    public CharLanguageModel? LanguageModel { get; init; }

    /// <summary>
    /// Weight of the language model's log probabilities against the recognizer's, at least 0. Default 0.8.
    /// Scores are centred on the model's average, so a typical word adds about zero and an unusual one subtracts; no single
    /// character subtracts more than one nat times this weight.
    /// </summary>
    public double LanguageModelWeight { get; init; } = 0.8;

    /// <summary>The user's words, or null for none.</summary>
    public Lexicon? Lexicon { get; init; }

    /// <summary>
    /// Log-domain bonus, at least 0, for a word that is spelled out in the <see cref="Lexicon"/> (half of it while the word is
    /// still a prefix of a listed one). A bonus, never a rule. Default 2.
    /// </summary>
    public double LexiconBoost { get; init; } = 2.0;

    /// <summary>Log-domain bonus per emitted character, any real number. Default 0; positive values favour longer readings.</summary>
    public double InsertionBonus { get; init; }

    internal Prepared Prepare(IReadOnlyList<string> charset)
    {
        Prepared? cached = _prepared;
        if (cached is not null && ReferenceEquals(cached.Charset, charset))
        {
            return cached;
        }

        var made = new Prepared(this, charset);
        _prepared = made;
        return made;
    }

    internal void Validate(OcrLimits limits)
    {
        if (!Enum.IsDefined(Mode))
        {
            throw new ArgumentException("Unknown decoding mode.");
        }

        if (Mode != CtcDecoding.Beam)
        {
            return;
        }

        if (BeamWidth < 1)
        {
            throw new ArgumentException("The beam width must be at least 1.");
        }

        if (BeamWidth > limits.MaxBeamWidth)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxBeamWidth), limits.MaxBeamWidth, BeamWidth);
        }

        if (!double.IsFinite(LanguageModelWeight) || LanguageModelWeight < 0 || !double.IsFinite(LexiconBoost) || LexiconBoost < 0
            || !double.IsFinite(InsertionBonus))
        {
            throw new ArgumentException("Weights must be finite, and the language model weight and lexicon boost at least 0.");
        }
    }
}
