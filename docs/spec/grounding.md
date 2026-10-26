# Grounding, text merge and read outputs (v1)

Behavior of `LlmClient.ReadAsync` in `RecursiveAI.OneOcr.Llm`. Everything here is deterministic given the model's answer
and the page model: no clock, no locale, no randomness. A port must reproduce it exactly.

## Read outputs

`LlmOutputs` flags: `Text`, `Markdown`, `Json`, `PageModel`. One model call produces every requested output.

The call asks for one JSON object with these members, present only when asked for:

| member | when | content |
|---|---|---|
| `pages` | `Text` or `PageModel` | array of strings, one per page image sent: the exact text of the page, lines separated by a newline |
| `markdown` | `Markdown` | array of strings, one per page image sent: the page as Markdown |
| `data` | `Json` | the caller's schema with every scalar replaced by `{ "value", "words", "derived" }` (below) |

The page images and the word list are the existing request: words are listed as `[id, text, left, top, right, bottom]`,
ids `w0`, `w1`, … in the order of the pages' word sequences, masked words left out.

Results: `Text` (merged), `LlmText` (the model's own), `Markdown` and `MarkdownFlags`, `Json` and `Fields`, `Pages`
(merged when `PageModel` was asked, otherwise the input pages), `Disagreements`, `Calls`, `Usage`, `CostUsd`,
`Diagnostics`. Pages are joined by a blank line (`\n\n`).

An answer that does not have the requested shape never throws on its own: a missing or short `pages`/`markdown` array
gives empty pages and a diagnostic; a missing `data` gives all-null values.

## Wrapped schema

The caller's schema is read into a tree: objects with `properties`, arrays with `items`, scalar leaves (string, number,
integer, boolean, enums). Supported shapes: `$ref` to `#/$defs/…` or `#/definitions/…` (inlined, at most 8 levels, at most
20,000 nodes), `anyOf` of exactly one type and `null`, type arrays containing `null`. Not supported (the call throws
`ArgumentException` before anything is sent): `oneOf`, `allOf`, `prefixItems`, objects without `properties`, arrays
without `items`.

Every object lists all its properties as required and forbids others. A leaf becomes

```json
{ "value": <the leaf schema, made nullable>, "words": ["w3","w4"], "derived": false }
```

`value` is `null` when the value is not printed. `words` are the ids of every word the value was read from. `derived` is true
only for values computed or inferred rather than printed. The plain JSON returned to the caller is rebuilt from the tree
and the answer; it has the caller's shape (missing values are `null`, arrays keep the model's items).

Paths are JSON Pointers: `/items/0/amount` (`~` is `~0`, `/` is `~1`).

## Grounding one value

Input: the value as text (strings as they are, numbers and booleans as their JSON text), the cited ids, the derived flag,
and all listed words with their calibrated confidence.

Steps, first match wins:

1. **Boolean**: state `Derived`, confidence 0.5, never boxed.
2. **Cited words**: look for the value in the cited words (ids sorted ascending, duplicates and unknown ids dropped; unknown
   ids are noted). A match of kind *exact* or *normalized* is taken: `Grounded` or `GroundedNormalized`.
3. **Page search**: look for the value in all listed words. Take the best match: exact before normalized before part-of-word
   before interleaved; among equals the one nearest a cited word (smallest word-index distance), then the first. The field
   is marked `CitationCorrected`, with a note saying whether nothing was cited or the cited words did not hold the value, and
   how many places matched. If the cited words held only a part-of-word or interleaved match and the page has no better one,
   the citation is kept and not marked corrected.
4. **Derived**: if the model said `derived`: for an amount, check whether the cited amounts reproduce it (below); state
   `Derived`, confidence 0.9 when reproduced, 0.3 otherwise. Derived values are never boxed.
5. **Ungrounded**: confidence 0. The note names the closest page text when one is within 40% edit distance (page of at most
   5,000 words, value of at most 64 folded characters).

### Matching a run of words

A *run* is consecutive words in the scope (the cited list, or the page). Strategies, all tried; their hits are pooled:

- **Text.** Fold both sides, then compare the run's folded concatenation with the value's folded text after trimming `. , ; :`
  from both ends. Fold = Unicode NFKC, typographic quotes to `'` and `"`, dashes and minus to `-`, lower case (invariant),
  every whitespace, format and control character removed. A value that is only punctuation is compared untrimmed. Runs
  grow word by word, up to 64 words, and stop at the first word where the concatenation can no longer be a prefix of the
  value. The hit is **exact** (state `Grounded`) when the run's text with whitespace removed equals the value with whitespace
  removed, case and punctuation included; otherwise **normalized**.
- **Amount.** If the value parses as an amount: single words and pairs of words parsed as amounts; equal decimal value and
  compatible currency (either side without currency is compatible; otherwise the sets of ISO codes must intersect). Normalized form:
  below.
- **Date.** If the value parses as a date: runs of 1 to 4 words (starting at a word with a digit or a month name) parsed as
  dates. Match when month and day are equal and the years are equal or either side has none. A year-less run is rejected
  when the next word would extend it into a date that has a year (`24 October` inside `24 October 2024`). Hit flags:
  `YearInferred` when the page has no year, `DateAmbiguous` when the page's reading of day/month was not unique.
- **Currency.** If the value is an ISO 4217 code from the table: any word that is that code or a symbol standing for it.
- **Interleaved.** Only when nothing else matched and the value has several words: each word of the value, folded and
  trimmed, equals a word in order with at most 3 other words between neighbours. Only the matching words are boxed.
- **Part of a word.** Only when nothing else matched and the folded value has at least 4 characters: one word contains it with
  no letter or digit directly before or after.

### Amounts

An amount is optional currency, optional sign, digits with separators, optional currency, optional sign. Accepted: a
currency symbol or ISO code at the start or end (`€`, `$`, `US$`, `£`, `¥`, `₹`, `R$`, `zł`, `kr`, `CHF`, …; `$` stands for
USD, CAD, AUD, NZD, MXN, SGD, HKD, ARS, CLP, COP); a leading `-`, `−`, `+`, a trailing `-`, or parentheses for negative;
digit groups separated by `.`, `,`, space, no-break space, narrow no-break space, thin space or `'`.

Decimal mark, given the page's **number style** (`PointDecimal` or `CommaDecimal`):

- both `.` and `,` present: the last one is the decimal mark;
- one kind, more than once: group marks (every group after the first has exactly three digits);
- one kind, once: group mark when exactly three digits follow it, one to three digits come before it and the number does not
  start with `0` (`1,234`, `1.234`); then it is decimal only if the style's mark is that character. In every other case
  (`12,5`, `0,125`, `1234,567`) it is the decimal mark;
- spaces and apostrophes are always group marks.

The **page number style** (`NumberStyle.Auto`) votes over listed words that look numeric: both marks present, last is the
decimal; one mark repeated: the other mark is the decimal; one mark once followed by one, two, or four or more digits: that mark
is the decimal. More comma votes than point votes: comma; otherwise point. A caller can pin the style.

A JSON number value is read as an invariant decimal. Canonical text: `.` decimal mark, no grouping, trailing zeros removed,
at least two decimals when it has any (`12.5` is `12.50`, `12` is `12`, `0.125` is `0.125`).

### Dates

Tokens are digit runs and letter runs; other characters (space `.` `,` `/` `-` and Unicode hyphens, no-break space) only
separate; any other character rejects. Ordinal suffixes (`st nd rd th er º`) and the filler words `of de del der the le den am on`
are ignored. At most three tokens remain, at most one letter token, which must be a month name or abbreviation in English,
German, French, Spanish, Italian, Portuguese, Dutch or Turkish (table in the source).

Readings: eight digits → year, month, day. Three numbers: first of four digits → year, month, day; otherwise day/month first
two numbers (1–2 digits) and a last number of two or four digits as the year. Two-digit years: 00–69 → 2000–2069, 70–99 →
1970–1999. If the first number is above 12 it is the day; if the second is above 12 it is the day; equal numbers read the
same either way; otherwise the **date order** decides, `DayFirst` or `MonthFirst`. With `DateOrder.Auto` the order is the
page's: among listed words of the shape `n.n.yyyy`, `n/n/yyyy`, `n-n-yyyy` with only one number above 12, the majority;
with no evidence both readings are kept and the field is flagged `DateAmbiguous`. With a month name: day and year
(`3 Oct 2026`, `Oct 3, 2026`, `2026 Oct 3`) or day only (year missing). A date must exist (`31.02.2026` is rejected;
29 February is accepted without a year). Canonical: `yyyy-MM-dd`, or `--MM-dd` without a year.

Times of day are not normalized (they match as text).

### Derived amounts

Over at most 16 cited words that parse as amounts: the value equals the sum of any subset of two or more; or `a − b`;
or `a × b` rounded to two decimals; or `a × b / 100` rounded to two decimals (percentages). Midpoints round to even.

### Confidence

`confidence = base × mean(word confidence)`, clamped to 0..1, where the mean runs over the boxed words' calibrated confidence
and `base` is the product of: exact 1.0, normalized 0.95, part of a word 0.9, interleaved 0.8; × 0.85 when the citation was
corrected; × 0.8 more when the page search found several places; × 0.7 when the year was inferred; × 0.8 when day and
month were ambiguous. Ungrounded 0; derived 0.9 or 0.3; boolean 0.5.

### Boxes

One box per line of backing words: consecutive words on the same page whose vertical overlap with the previous box is at
least half the smaller height are united.

## Re-ask

After the first answer, fields in state `Ungrounded` or `RoleConflict` (not `Derived`) are re-asked once if `Reask` is true (default). The
second call carries the same images and word list and a second delimited block `<<<CHECK tag … CHECK tag>>>` with one JSON
array per field (at most 64): `[path, value given, text of the cited words, closest page text or null]`; for a role conflict the last item is the value the local model reads for that path, when it has one. It asks for the
data member only, in full. For each re-asked path present in the new answer, the new leaf replaces the old and is grounded
again (`Reasked` true). A path absent from the new answer, or null there, is withdrawn: its value is `null`, it is not in
`Fields`, and a diagnostic says so. Other fields keep the first answer.

The second call goes through the same client as every call: spend caps, per-call bounds, concurrency and retries apply. If it
is refused (`OcrBudgetException`) or fails (`OcrProviderException`), the first results stand and a diagnostic says why.
There is never a third call.

`UngroundedPolicy` then applies to fields still `Ungrounded` or `RoleConflict`: `Flag` keeps the value in the JSON (the field says it is
ungrounded); `Null` writes `null`; `Throw` throws `OcrProviderException` with `LlmFailure.Ungrounded`, naming up to eight paths.

## Text merge (`ExactText`)

Per page sent: the OCR words (the page's word sequence) and the model's tokens (split on whitespace) are aligned.

**Keys.** A word's key is its fold with `. , ; :` trimmed from both ends.

**Alignment.** Global alignment over keys, minimizing cost: a gap costs 1000; a pairing costs 0 for equal keys, otherwise
`400 + floor(1600 × d / max(len))` with `d` the Levenshtein distance of the keys; ties prefer pairing, then dropping the
OCR word, then the model's word. At most 2^22 cells are computed in full; larger pages use a band of half-width
`max(256, |n − m| + 64)` around the diagonal; if that still needs more than 64M cells the page keeps its local text and a
diagnostic says so.

**Merge**, per aligned pair of OCR word `o` and model token `t`:

- same text (ordinal): the word, source of the OCR word, confidence unchanged;
- different text and calibrated `o.Confidence ≥ ConfidenceThreshold`: the OCR word wins (source of `o`);
- different text otherwise: the model's text, source `Llm`, confidence 0.5, box of `o`. No calibrated estimate exists for
  model words; 0.5 marks "uncertain".

Only the model: kept with source `Llm`, confidence 0.5 and a zero-width box at the right edge of the previous OCR word (the
left edge of the next one at the start of a page) unless `KeepLlmOnlyWords` is false. Only OCR: kept (after the previous
word) when confident, dropped otherwise. Each of these, and each substitution, is a `TextDisagreement` with the choice made;
`Start` and `Length` locate the chosen word in `Text` (a dropped word has length 0).

Text is rebuilt in the model's order with the model's separators; OCR-only words that are kept are inserted after the
previous word with one space.

**Merged page model.** `Pages[i]` is the input page with each word replaced by its merged word (`Source`, `Confidence`;
characters are kept only where the text is unchanged), dropped words removed, model-only words added to the line of the previous word (cells
unchanged), empty lines removed, table cells and key–value pairs remapped to the new word indices (removed words leave
them; pairs left with no key or no value go). Blocks and regions are kept. A diagnostic is added.

## Markdown flags

The model's Markdown per page is split on whitespace. In each piece, leading `# > * _ | ~ ` ` and trailing `* _ | ~ ` ` are
syntax; pieces made only of `- + | : # > * _ ~ ` =` are not words. The remaining words are aligned with the OCR words as
above. A flag is raised for a pair whose keys differ (so case and edge punctuation do not flag), for a model-only word, and
for a confident OCR word the Markdown lacks (zero length, at the position of the next word). `Start`/`Length` are offsets into
`Markdown` covering the word without its syntax characters. `MarkdownWithFlags` wraps flagged spans in `⟦ ⟧` by default.

## Routing

- `LlmOnly` (default): the model is called for every requested output.
- `LocalOnly`: the model is never called, even if configured or when offline. `Text` and `PageModel` are the local text and
  pages; `Markdown` is not produced (diagnostic); `Json` needs an `ILocalFieldExtractor` (none ships) or the call throws
  `ArgumentException`.
- `LocalFirst`: `Json` is routed per field when a local extractor is set (`LlmReadOptions.LocalExtractor`; `DocumentFieldExtractor` wraps a
  `DocumentExtractor`, the shipped invoice and receipt model or a custom one). `Markdown` always needs the model. For `Text` and
  `PageModel` only pages with a word below the threshold are sent; the others keep their local text. No page to send means no call.

### Per-field `LocalFirst`

The extractor returns the schema's values as JSON plus a list of fields (schema path, value, words, confidence). Then:

1. A scalar is **local** when the list has it at `ConfidenceThreshold` or above, in state `Grounded`/`GroundedNormalized`, with a
   non-null value in the JSON. A scalar the extractor lacks, is unsure of, or reads as null is **missing** and goes to the model.
2. An array is local only **as a whole**: the extractor has at least one element and every value of every element is local.
   Otherwise the whole array goes to the model, because the model and the local reader may disagree on how many elements there are.
3. The model is asked for the **reduced schema**: the same schema with the local scalars and arrays removed (objects left empty are
   removed too). No local values are sent as hints. If nothing is left, no call is made, nothing is sent and nothing is spent.
4. The result merges both: local values keep their fields with `Source = Ocr` (the model labelled words on the page; the words' own
   source is on the page model), model values are grounded as usual with `Source = Llm`. In the JSON, a local array is the
   extractor's, a model array the model's. The estimate, the cost check and `SpentUsd` are those of the reduced request, because that is
   the request that is built. Diagnostics say how many values were kept.
5. An extractor that lists no fields (only a JSON and a `MinConfidence`) keeps the earlier whole-document rule: its result is used when
   `MinConfidence` reaches the threshold, otherwise the model answers every value.

`LocalOnly` returns the extractor's JSON and fields and never calls the model. `Offline` makes every call throw, so a `LocalFirst`
read that needs the model throws before anything is sent, while one that is fully local makes no call and does not throw.

`DocumentFieldExtractor` matches a schema property to a class of the model: first by the property's name equal to a class name (case
and punctuation ignored), then by a class named as a whole token in the property's description (`BT-112`), then by a short list of common
words for EN 16931 terms and receipt fields (`total`, `invoiceNumber`, `vendor`, `iban`, `qty` and so on). Where an invoice and a receipt
class share a word, the one the model found is used. An array of objects takes the model's line items, or its tax rows when its name
contains `tax` or `vat`; element *i* is group *i*. A property without a class, or whose class was not found, is missing. A number-typed
property is written as a JSON number from the field's parsed number (missing when the text did not parse). The field's confidence is the
model's mean tag probability, not a calibrated one.

## Role check

Grounding checks that a value is on the page, not that it is in the right role. When a local extractor is set (any routing except
`LocalOnly`; `LlmReadOptions.RoleCheck`, default true), every model value in state `Grounded` or `GroundedNormalized` is also checked against
the local reading, using only fields the local model holds at `ConfidenceThreshold` or above:

- one of the cited words is a word of a local field of **another path** (or of a field outside the schema, `LocalFieldResult.OtherFields`); or
- the local model reads **this path** from other words (none shared with the citation) and its value, compared after normalization,
  is different.

Only a property the local model has a class for is judged (`LocalFieldResult.MappedPaths`: the schema paths, array indices left out, that
some class of the model answers to, whether or not it found a value on this page). For a property with no class the local model says
nothing about what it means, so its cited words belonging to another class is no contradiction and the value stays as grounding left it.
A local extractor that gives no set is taken to have a class for every property. `DocumentFieldExtractor` fills the set with the
properties its name, description and alias matching maps to a class.

The field then gets state **`RoleConflict`** (a state of its own rather than a flag on `Grounded`, so code that trusts `Grounded`
cannot overlook it). It keeps its words and boxes, has confidence 0 and a note saying what the local model reads. It counts like
`Ungrounded` for the re-ask (the hint is the local value), for the check after the re-ask, and for `UngroundedPolicy`. The check costs
no call. A value the local model does not know (a missing field, a low-confidence one) is not checked: no local reading is not a
conflict. Derived values and values without cited words are not checked.

## Prompt injection

Page text and the model's own earlier answer are untrusted. They reach the model only as JSON-escaped array elements, one
per line, between marker lines whose tag is the SHA-256 of the block (so a page cannot forge a closing marker), with a
system prompt saying to treat them as data; the model has no tools and answers in a fixed schema. Nothing from a model answer is executed or
used as a path, id or URL except as a lookup key into the word list.

What grounding catches: a value the page does not contain (invented, leaked from the instructions, an injected number that
is not printed), a citation that points elsewhere (corrected and flagged), a computed value that nothing reproduces
(`Derived`, 0.3). What it cannot catch: the model choosing a different number or text that is really printed on the page, including
hostile text printed on it; and a printed value in the wrong role (a vendor name in the total field). Those are grounded because
they are on the page; with a local extractor the role check above catches them whenever the local model confidently reads the field
(the `printed-other-number`, `tiny-text-read` and `list-spoof` cases of the injection set become `RoleConflict`); only the cases the local
model is unsure of remain. Mitigations are the user's: redaction, a schema with descriptions, and checking boxes and confidences for
fields that matter. The model-answer set in `tests/OneOcr.Tests/LlmInjectionTests.cs` records, per attack, what grounding
reports.
