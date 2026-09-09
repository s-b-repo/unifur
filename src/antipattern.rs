//! Negative supervision for code: anti-pattern rules, categories, and the
//! per-token labels a penalized loss reads (roadmap Phase 24).
//!
//! Ordinary next-token training has one signal: *make the corpus more likely*.
//! Every `except: pass` in the training data is therefore a lesson in writing
//! `except: pass`. This module is the other half of the signal. A rule set
//! names the idioms that should **not** be learned, sorts them into categories,
//! and labels every token that belongs to one. The loss in
//! [`crate::lm::LanguageModel::next_token_loss_penalized`] then charges the
//! model for the probability it assigns to those tokens instead of rewarding it.
//!
//! # Context and body
//!
//! A rule has two patterns. The **context** is matched but not penalized; the
//! **body** is matched *and* penalized. For
//!
//! ```text
//! except:
//!     pass
//! ```
//!
//! the context is `except\b[^:\n]*:\s*` and the body is `pass\b`. Nothing is
//! wrong with `except:`; the failure is choosing `pass` after it. Labeling the
//! whole match would push the model away from writing `except` at all, which
//! is the opposite of teaching it to handle errors.
//!
//! # The pattern language
//!
//! No regex crate is used — this crate is deliberately dependency-light — so
//! patterns are matched by a small backtracking engine over a subset of the
//! usual syntax:
//!
//! | Syntax | Meaning |
//! |---|---|
//! | `a`, `\(`, `\\` | a literal byte (escape the metacharacters `.[]()\|{}*+?^$`) |
//! | `.` | any byte except newline |
//! | `\s` `\w` `\d` | whitespace, `[A-Za-z0-9_]`, `[0-9]`; upper case negates |
//! | `[abc]`, `[a-z]`, `[^)]` | a class, a range, a negated class |
//! | `\b`, `$` | a word boundary, the end of a line or of the text |
//! | `*`, `+`, `?`, `{n}`, `{n,m}` | greedy quantifiers on the preceding atom |
//!
//! Alternation and groups are **not** supported; an unescaped `(`, `)` or `|`
//! is a parse error rather than a silent literal. Two rules are cheaper than
//! one wrong one.
//!
//! # Tokens, not characters
//!
//! Matching runs over token ids, not text. With the byte-level tokenizer a
//! byte's id *is* its value, so a pattern written against text applies
//! unchanged — and a special token (`<bos>`, `<eos>`, `<pad>`) matches
//! nothing, so no rule can span two documents.
//!
//! # No Burn dependency
//!
//! Like [`crate::expert_index`], this module is plain data plus `serde`. A rule
//! set is a JSON file a reviewer can read, and the labeler can be run by a tool
//! that never loads a model.

use anyhow::{bail, Context as _, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The label of a token no rule matched.
pub const CLEAN: u8 = 0;

/// Labels are one byte, and 0 is reserved.
pub const MAX_CATEGORIES: usize = 255;

/// Manifest schema version.
pub const MANIFEST_VERSION: u32 = 1;

// ------------------------------------------------------------------ rules --

/// A kind of bad code, with the penalty weight its tokens carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Category {
    pub name: String,
    /// Multiplier on the unlikelihood term for tokens in this category.
    /// `1.0` is full strength; `0.0` labels the tokens but charges nothing.
    pub weight: f32,
    #[serde(default)]
    pub description: String,
}

/// One idiom to unlearn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    /// Must name an entry in [`RuleSet::categories`].
    pub category: String,
    #[serde(default)]
    pub language: String,
    /// Matched, not penalized. May be empty.
    #[serde(default)]
    pub context: String,
    /// Matched **and** penalized. Must consume at least one token.
    pub body: String,
    /// Texts the rule must match. Checked by [`RuleSet::validate`].
    #[serde(default)]
    pub examples: Vec<String>,
    /// Texts the rule must **not** match. Checked by [`RuleSet::validate`].
    #[serde(default)]
    pub counterexamples: Vec<String>,
}

/// Categories plus the rules that populate them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleSet {
    pub categories: Vec<Category>,
    pub rules: Vec<Rule>,
}

impl RuleSet {
    /// The rules this crate ships. See the module documentation for the
    /// categories; every rule carries the examples it is certified against.
    pub fn builtin() -> Self {
        builtin_rules()
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let set: Self = serde_json::from_str(json).context("parse rule set")?;
        set.validate()?;
        Ok(set)
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serialize rule set")
    }

    pub fn read(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        Self::from_json(&json).with_context(|| format!("in {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_json()?)
            .with_context(|| format!("write {}", path.display()))
    }

    /// Position of `name` in `categories`, if present.
    pub fn category_index(&self, name: &str) -> Option<usize> {
        self.categories.iter().position(|c| c.name == name)
    }

    /// Structural validation **and** the self-test every rule carries.
    ///
    /// A rule set whose examples do not match is refused rather than loaded:
    /// a rule that matches nothing labels nothing, and the training run would
    /// report a penalty of zero as if the corpus were clean.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.categories.is_empty(), "a rule set needs at least one category");
        anyhow::ensure!(
            self.categories.len() <= MAX_CATEGORIES,
            "{} categories, but labels are one byte with 0 reserved (max {MAX_CATEGORIES})",
            self.categories.len()
        );
        for (i, category) in self.categories.iter().enumerate() {
            anyhow::ensure!(!category.name.is_empty(), "category {i} has an empty name");
            anyhow::ensure!(
                category.weight.is_finite() && category.weight >= 0.0,
                "category '{}' has weight {}, which must be finite and >= 0",
                category.name,
                category.weight
            );
            if self.categories[..i].iter().any(|c| c.name == category.name) {
                bail!("category '{}' is declared twice", category.name);
            }
        }
        anyhow::ensure!(!self.rules.is_empty(), "a rule set needs at least one rule");
        for (i, rule) in self.rules.iter().enumerate() {
            anyhow::ensure!(!rule.name.is_empty(), "rule {i} has an empty name");
            if self.rules[..i].iter().any(|r| r.name == rule.name) {
                bail!("rule '{}' is declared twice", rule.name);
            }
            anyhow::ensure!(
                self.category_index(&rule.category).is_some(),
                "rule '{}' names unknown category '{}'",
                rule.name,
                rule.category
            );
            // Flattened rather than chained: `validate` is what a CLI prints,
            // and the reason must survive `Display`, not only `{:#}`.
            let compiled = CompiledRule::compile(rule)
                .map_err(|err| anyhow::anyhow!("rule '{}': {err:#}", rule.name))?;
            for example in &rule.examples {
                let tokens = text_tokens(example);
                anyhow::ensure!(
                    compiled.first_match(&tokens).is_some(),
                    "rule '{}' does not match its own example {example:?}",
                    rule.name
                );
            }
            for counter in &rule.counterexamples {
                let tokens = text_tokens(counter);
                if let Some(span) = compiled.first_match(&tokens) {
                    bail!(
                        "rule '{}' matches its counterexample {counter:?} at bytes {}..{}",
                        rule.name,
                        span.0,
                        span.1
                    );
                }
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------- patterns --

#[derive(Debug, Clone)]
enum Atom {
    Literal(u16),
    /// Any byte but newline.
    Any,
    Class { negated: bool, set: Box<[bool; 256]> },
    /// Zero width: `is_word` differs on the two sides.
    WordBoundary,
    /// Zero width: end of text or just before a newline.
    LineEnd,
    /// Zero width: records where the body starts.
    Mark,
}

impl Atom {
    fn is_zero_width(&self) -> bool {
        matches!(self, Atom::WordBoundary | Atom::LineEnd | Atom::Mark)
    }

    fn matches(&self, token: u16) -> bool {
        // A special token is not a byte and matches nothing, so no rule can
        // reach across a document boundary.
        if token >= 256 {
            return false;
        }
        match self {
            Atom::Literal(x) => token == *x,
            Atom::Any => token != u16::from(b'\n'),
            Atom::Class { negated, set } => set[token as usize] != *negated,
            Atom::WordBoundary | Atom::LineEnd | Atom::Mark => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Quant {
    min: usize,
    max: usize,
}

const ONE: Quant = Quant { min: 1, max: 1 };

/// A compiled pattern: atoms with quantifiers, matched by backtracking.
#[derive(Debug, Clone)]
pub struct Pattern {
    atoms: Vec<(Atom, Quant)>,
}

fn class_of(pred: impl Fn(u8) -> bool) -> Box<[bool; 256]> {
    let mut set = Box::new([false; 256]);
    for (b, slot) in set.iter_mut().enumerate() {
        *slot = pred(b as u8);
    }
    set
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `\x` outside a class. Returns the atom and whether it may take a quantifier.
fn escape_atom(c: u8) -> Result<Atom> {
    Ok(match c {
        b's' => Atom::Class { negated: false, set: class_of(is_space) },
        b'S' => Atom::Class { negated: true, set: class_of(is_space) },
        b'w' => Atom::Class { negated: false, set: class_of(is_word_byte) },
        b'W' => Atom::Class { negated: true, set: class_of(is_word_byte) },
        b'd' => Atom::Class { negated: false, set: class_of(|b| b.is_ascii_digit()) },
        b'D' => Atom::Class { negated: true, set: class_of(|b| b.is_ascii_digit()) },
        b'b' => Atom::WordBoundary,
        b'n' => Atom::Literal(u16::from(b'\n')),
        b't' => Atom::Literal(u16::from(b'\t')),
        b'r' => Atom::Literal(u16::from(b'\r')),
        b'.' | b'[' | b']' | b'(' | b')' | b'\\' | b'|' | b'{' | b'}' | b'*' | b'+' | b'?'
        | b'^' | b'$' | b'/' | b'-' | b'#' | b'"' | b'\'' => Atom::Literal(u16::from(c)),
        other => bail!("unknown escape '\\{}'", char::from(other)),
    })
}

/// Fold an escape into a class under construction.
fn escape_into_class(c: u8, set: &mut [bool; 256]) -> Result<Option<u8>> {
    // Returns Some(byte) for a single literal (so a range can use it), None
    // for a multi-byte shorthand already folded in.
    match c {
        b's' | b'w' | b'd' => {
            let shorthand = escape_atom(c)?;
            if let Atom::Class { set: inner, .. } = shorthand {
                for (slot, on) in set.iter_mut().zip(inner.iter()) {
                    *slot |= *on;
                }
            }
            Ok(None)
        }
        b'n' => {
            set[usize::from(b'\n')] = true;
            Ok(Some(b'\n'))
        }
        b't' => {
            set[usize::from(b'\t')] = true;
            Ok(Some(b'\t'))
        }
        b'r' => {
            set[usize::from(b'\r')] = true;
            Ok(Some(b'\r'))
        }
        b']' | b'[' | b'\\' | b'-' | b'^' | b'.' | b'(' | b')' | b'|' | b'{' | b'}' | b'*'
        | b'+' | b'?' | b'$' | b'/' | b'#' | b'"' | b'\'' => {
            set[usize::from(c)] = true;
            Ok(Some(c))
        }
        other => bail!("unknown escape '\\{}' inside a class", char::from(other)),
    }
}

impl Pattern {
    /// Compile `source`. See the module documentation for the syntax.
    pub fn parse(source: &str) -> Result<Self> {
        let bytes = source.as_bytes();
        let mut atoms: Vec<(Atom, Quant)> = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            i += 1;
            let atom = match c {
                b'\\' => {
                    let esc = *bytes.get(i).context("pattern ends in a lone backslash")?;
                    i += 1;
                    escape_atom(esc)?
                }
                b'.' => Atom::Any,
                b'$' => Atom::LineEnd,
                b'[' => {
                    let (atom, next) = parse_class(bytes, i)?;
                    i = next;
                    atom
                }
                b'(' | b')' | b'|' => bail!(
                    "'{}' at byte {}: groups and alternation are not supported; escape it or split the rule",
                    char::from(c),
                    i - 1
                ),
                b'*' | b'+' | b'?' | b'{' => {
                    bail!("quantifier '{}' at byte {} has nothing to repeat", char::from(c), i - 1)
                }
                b']' | b'}' => bail!("unmatched '{}' at byte {}", char::from(c), i - 1),
                other => Atom::Literal(u16::from(other)),
            };

            // A quantifier, if one follows.
            let mut quant = ONE;
            if let Some(&q) = bytes.get(i) {
                let parsed = match q {
                    b'*' => Some((Quant { min: 0, max: usize::MAX }, i + 1)),
                    b'+' => Some((Quant { min: 1, max: usize::MAX }, i + 1)),
                    b'?' => Some((Quant { min: 0, max: 1 }, i + 1)),
                    b'{' => Some(parse_braces(bytes, i + 1)?),
                    _ => None,
                };
                if let Some((q, next)) = parsed {
                    anyhow::ensure!(
                        !atom.is_zero_width(),
                        "a quantifier at byte {i} cannot apply to a zero-width assertion"
                    );
                    quant = q;
                    i = next;
                }
            }
            atoms.push((atom, quant));
        }
        Ok(Self { atoms })
    }

    /// Whether some atom must consume at least one token.
    pub fn consumes(&self) -> bool {
        self.atoms.iter().any(|(a, q)| !a.is_zero_width() && q.min >= 1)
    }

    /// Anchored match at `at`. Returns the end of the match and, if a
    /// [`Atom::Mark`] was crossed, its position.
    fn match_at(&self, text: &[u16], at: usize) -> Option<(usize, Option<usize>)> {
        self.go(text, 0, at, None)
    }

    fn go(
        &self,
        text: &[u16],
        idx: usize,
        pos: usize,
        mark: Option<usize>,
    ) -> Option<(usize, Option<usize>)> {
        let Some((atom, quant)) = self.atoms.get(idx) else {
            return Some((pos, mark));
        };
        match atom {
            Atom::Mark => self.go(text, idx + 1, pos, Some(pos)),
            Atom::WordBoundary => {
                let before = pos > 0 && is_word_token(text[pos - 1]);
                let after = pos < text.len() && is_word_token(text[pos]);
                (before != after).then(|| self.go(text, idx + 1, pos, mark)).flatten()
            }
            Atom::LineEnd => {
                let at_end = pos >= text.len() || text[pos] == u16::from(b'\n');
                at_end.then(|| self.go(text, idx + 1, pos, mark)).flatten()
            }
            _ => {
                // Greedy: take as many as possible, then give back one at a
                // time until the rest of the pattern fits.
                let mut n = 0;
                while n < quant.max && pos + n < text.len() && atom.matches(text[pos + n]) {
                    n += 1;
                }
                if n < quant.min {
                    return None;
                }
                (quant.min..=n)
                    .rev()
                    .find_map(|k| self.go(text, idx + 1, pos + k, mark))
            }
        }
    }
}

fn is_word_token(token: u16) -> bool {
    token < 256 && is_word_byte(token as u8)
}

/// Parse `[...]` starting just after the `[`. Returns the atom and the index
/// after the closing `]`.
fn parse_class(bytes: &[u8], mut i: usize) -> Result<(Atom, usize)> {
    let open = i - 1;
    let mut set = [false; 256];
    let negated = bytes.get(i) == Some(&b'^');
    if negated {
        i += 1;
    }
    let mut prev: Option<u8> = None;
    let mut first = true;
    loop {
        let Some(&c) = bytes.get(i) else {
            bail!("class opened at byte {open} is never closed");
        };
        i += 1;
        match c {
            b']' if !first => break,
            b'\\' => {
                let esc = *bytes.get(i).context("class ends in a lone backslash")?;
                i += 1;
                prev = escape_into_class(esc, &mut set)?;
            }
            b'-' if prev.is_some() && bytes.get(i).is_some_and(|n| *n != b']') => {
                // A range `a-z`, where `z` may itself be escaped.
                let lo = prev.take().expect("checked");
                let mut hi = bytes[i];
                i += 1;
                if hi == b'\\' {
                    let esc = *bytes.get(i).context("class ends in a lone backslash")?;
                    i += 1;
                    hi = escape_into_class(esc, &mut set)?
                        .with_context(|| format!("'\\{}' cannot end a range", char::from(esc)))?;
                }
                anyhow::ensure!(lo <= hi, "range {}-{} is backwards", char::from(lo), char::from(hi));
                for b in lo..=hi {
                    set[usize::from(b)] = true;
                }
            }
            other => {
                set[usize::from(other)] = true;
                prev = Some(other);
            }
        }
        first = false;
    }
    Ok((Atom::Class { negated, set: Box::new(set) }, i))
}

/// Parse `n}` or `n,m}` starting just after the `{`.
fn parse_braces(bytes: &[u8], start: usize) -> Result<(Quant, usize)> {
    let close = bytes[start..]
        .iter()
        .position(|b| *b == b'}')
        .with_context(|| format!("'{{' at byte {} is never closed", start - 1))?;
    let inner = std::str::from_utf8(&bytes[start..start + close]).context("brace quantifier")?;
    let (min, max) = match inner.split_once(',') {
        Some((lo, hi)) => {
            let lo: usize = lo.trim().parse().with_context(|| format!("bad quantifier {{{inner}}}"))?;
            let hi = if hi.trim().is_empty() {
                usize::MAX
            } else {
                hi.trim().parse().with_context(|| format!("bad quantifier {{{inner}}}"))?
            };
            (lo, hi)
        }
        None => {
            let n: usize = inner.trim().parse().with_context(|| format!("bad quantifier {{{inner}}}"))?;
            (n, n)
        }
    };
    anyhow::ensure!(min <= max, "quantifier {{{inner}}} has min > max");
    Ok((Quant { min, max }, start + close + 1))
}

// ---------------------------------------------------------------- labeler --

#[derive(Debug, Clone)]
struct CompiledRule {
    pattern: Pattern,
    /// If the pattern must open with this byte, positions holding anything
    /// else are skipped without entering the matcher.
    first: Option<u16>,
}

impl CompiledRule {
    fn compile(rule: &Rule) -> Result<Self> {
        let context = Pattern::parse(&rule.context).context("context pattern")?;
        let body = Pattern::parse(&rule.body).context("body pattern")?;
        anyhow::ensure!(
            body.consumes(),
            "body {:?} can match the empty string, so it would label nothing",
            rule.body
        );
        let mut atoms = context.atoms;
        atoms.push((Atom::Mark, ONE));
        atoms.extend(body.atoms);
        let first = match atoms.first() {
            Some((Atom::Literal(b), q)) if q.min >= 1 => Some(*b),
            _ => None,
        };
        Ok(Self { pattern: Pattern { atoms }, first })
    }

    /// The body span of the match anchored at `at`, if any.
    fn body_at(&self, text: &[u16], at: usize) -> Option<(usize, usize)> {
        if let Some(first) = self.first {
            if text.get(at) != Some(&first) {
                return None;
            }
        }
        let (end, mark) = self.pattern.match_at(text, at)?;
        let start = mark.expect("every compiled rule carries a mark");
        (end > start).then_some((start, end))
    }

    fn first_match(&self, text: &[u16]) -> Option<(usize, usize)> {
        (0..=text.len()).find_map(|at| self.body_at(text, at))
    }
}

/// One penalized region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// First penalized token.
    pub start: usize,
    /// One past the last penalized token.
    pub end: usize,
    /// Index into [`Labeler::rule_names`].
    pub rule: usize,
    /// The label byte: `1 + ` the category's index.
    pub label: u8,
}

/// Compiled rules, ready to label token sequences.
#[derive(Debug, Clone)]
pub struct Labeler {
    set: RuleSet,
    compiled: Vec<CompiledRule>,
    /// Per rule, the label byte its category maps to.
    labels: Vec<u8>,
}

/// Text as token ids, which for bytes are the bytes themselves.
pub fn text_tokens(text: &str) -> Vec<u16> {
    text.as_bytes().iter().map(|b| u16::from(*b)).collect()
}

impl Labeler {
    /// Compile and validate `set`.
    pub fn new(set: RuleSet) -> Result<Self> {
        set.validate()?;
        let compiled = set
            .rules
            .iter()
            .map(CompiledRule::compile)
            .collect::<Result<Vec<_>>>()?;
        let labels = set
            .rules
            .iter()
            .map(|r| {
                let index = set.category_index(&r.category).expect("validated");
                u8::try_from(index + 1).expect("validated against MAX_CATEGORIES")
            })
            .collect();
        Ok(Self { set, compiled, labels })
    }

    /// The shipped rule set, compiled.
    pub fn builtin() -> Self {
        Self::new(RuleSet::builtin()).expect("the built-in rule set validates")
    }

    pub fn rule_set(&self) -> &RuleSet {
        &self.set
    }

    pub fn categories(&self) -> &[Category] {
        &self.set.categories
    }

    pub fn rule_names(&self) -> Vec<&str> {
        self.set.rules.iter().map(|r| r.name.as_str()).collect()
    }

    /// The category a label byte denotes, if any.
    pub fn category_of(&self, label: u8) -> Option<&Category> {
        if label == CLEAN {
            None
        } else {
            self.set.categories.get(usize::from(label) - 1)
        }
    }

    /// Label byte -> penalty weight. Index 0 (clean) is `0.0`.
    pub fn weight_table(&self) -> [f32; 256] {
        weight_table(&self.set.categories)
    }

    /// Every body span in `tokens`, in position order then rule order.
    ///
    /// Overlapping spans from different rules are all reported: a `catch`
    /// that is both empty and catches `Throwable` is two findings.
    pub fn scan(&self, tokens: &[u16]) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut last: Vec<Option<(usize, usize)>> = vec![None; self.compiled.len()];
        for at in 0..=tokens.len() {
            for (rule, compiled) in self.compiled.iter().enumerate() {
                if let Some(body) = compiled.body_at(tokens, at) {
                    // A context that starts with something optional can match
                    // from several positions onto the same body; report it once.
                    if last[rule] == Some(body) {
                        continue;
                    }
                    last[rule] = Some(body);
                    spans.push(Span { start: body.0, end: body.1, rule, label: self.labels[rule] });
                }
            }
        }
        spans
    }

    /// One label per token: [`CLEAN`] or `1 + category index`.
    ///
    /// Where spans overlap the **first** one in scan order wins, so a token
    /// carries one category and the corpus can be read back without the rule
    /// set. Use [`Self::scan`] for the full list of findings.
    pub fn label(&self, tokens: &[u16]) -> Vec<u8> {
        self.label_with_spans(tokens).0
    }

    pub fn label_with_spans(&self, tokens: &[u16]) -> (Vec<u8>, Vec<Span>) {
        let spans = self.scan(tokens);
        let mut labels = vec![CLEAN; tokens.len()];
        for span in &spans {
            for slot in &mut labels[span.start..span.end] {
                if *slot == CLEAN {
                    *slot = span.label;
                }
            }
        }
        (labels, spans)
    }

    /// Convenience for text: bytes are their own token ids.
    pub fn label_text(&self, text: &str) -> Vec<u8> {
        self.label(&text_tokens(text))
    }

    /// Summarize a labeling for the manifest written next to a corpus.
    pub fn manifest(&self, labels: &[u8], spans: &[Span]) -> LabelManifest {
        let mut categories: Vec<CategoryCount> = self
            .set
            .categories
            .iter()
            .map(|c| CategoryCount {
                name: c.name.clone(),
                weight: c.weight,
                description: c.description.clone(),
                tokens: 0,
                spans: 0,
            })
            .collect();
        for label in labels {
            if *label != CLEAN {
                categories[usize::from(*label) - 1].tokens += 1;
            }
        }
        let mut rules: Vec<RuleCount> = self
            .set
            .rules
            .iter()
            .map(|r| RuleCount {
                name: r.name.clone(),
                category: r.category.clone(),
                language: r.language.clone(),
                spans: 0,
            })
            .collect();
        for span in spans {
            rules[span.rule].spans += 1;
            categories[usize::from(span.label) - 1].spans += 1;
        }
        LabelManifest {
            format_version: MANIFEST_VERSION,
            tokens: labels.len(),
            labeled_tokens: labels.iter().filter(|l| **l != CLEAN).count(),
            categories,
            rules,
        }
    }

    /// Human-readable findings for `text`, one line per span with the line
    /// number and the offending source.
    pub fn report(&self, text: &str) -> String {
        let tokens = text_tokens(text);
        let spans = self.scan(&tokens);
        let names = self.rule_names();
        let mut out = String::new();
        for span in &spans {
            let line = text.as_bytes()[..span.start].iter().filter(|b| **b == b'\n').count() + 1;
            let category = self.category_of(span.label).map_or("?", |c| c.name.as_str());
            let snippet = String::from_utf8_lossy(&text.as_bytes()[span.start..span.end]);
            out.push_str(&format!(
                "line {line}: [{category}] {} -> {:?}\n",
                names[span.rule],
                snippet.as_ref()
            ));
        }
        out
    }
}

fn weight_table(categories: &[Category]) -> [f32; 256] {
    let mut table = [0.0f32; 256];
    for (i, c) in categories.iter().enumerate() {
        table[i + 1] = c.weight;
    }
    table
}

// --------------------------------------------------------------- manifest --

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryCount {
    pub name: String,
    pub weight: f32,
    #[serde(default)]
    pub description: String,
    /// Tokens carrying this label after overlap resolution.
    pub tokens: usize,
    /// Spans found, before overlap resolution.
    pub spans: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleCount {
    pub name: String,
    pub category: String,
    #[serde(default)]
    pub language: String,
    pub spans: usize,
}

/// What a `.labels` sidecar means: the category each label byte denotes, its
/// weight, and how much of the corpus each rule flagged.
///
/// Written as JSON next to the labels so the corpus can be trained on by a
/// build that has never seen the rule set that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelManifest {
    pub format_version: u32,
    pub tokens: usize,
    pub labeled_tokens: usize,
    /// In label order: `categories[k]` is what label byte `k + 1` means.
    pub categories: Vec<CategoryCount>,
    pub rules: Vec<RuleCount>,
}

impl LabelManifest {
    pub fn read(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&json)
            .with_context(|| format!("parse {}", path.display()))?;
        anyhow::ensure!(
            manifest.format_version == MANIFEST_VERSION,
            "{} is manifest version {}, this build reads {MANIFEST_VERSION}",
            path.display(),
            manifest.format_version
        );
        anyhow::ensure!(
            manifest.categories.len() <= MAX_CATEGORIES,
            "{} declares {} categories (max {MAX_CATEGORIES})",
            path.display(),
            manifest.categories.len()
        );
        Ok(manifest)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("serialize manifest")?;
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))
    }

    /// Label byte -> penalty weight, from the manifest's own category weights.
    pub fn weight_table(&self) -> [f32; 256] {
        let mut table = [0.0f32; 256];
        for (i, c) in self.categories.iter().enumerate() {
            table[i + 1] = c.weight;
        }
        table
    }

    /// Fraction of tokens carrying any label.
    pub fn labeled_fraction(&self) -> f64 {
        if self.tokens == 0 {
            0.0
        } else {
            self.labeled_tokens as f64 / self.tokens as f64
        }
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "{} tokens | {} labeled ({:.3}%)\n",
            self.tokens,
            self.labeled_tokens,
            100.0 * self.labeled_fraction()
        );
        for (i, c) in self.categories.iter().enumerate() {
            out.push_str(&format!(
                "  [{}] {:<24} weight {:<4} {:>9} tokens {:>7} spans\n",
                i + 1,
                c.name,
                c.weight,
                c.tokens,
                c.spans
            ));
        }
        for r in self.rules.iter().filter(|r| r.spans > 0) {
            out.push_str(&format!("      {:<32} {:<10} {:>7} spans\n", r.name, r.language, r.spans));
        }
        out
    }
}

// ---------------------------------------------------------------- builtin --

fn rule(
    name: &str,
    category: &str,
    language: &str,
    context: &str,
    body: &str,
    examples: &[&str],
    counterexamples: &[&str],
) -> Rule {
    Rule {
        name: name.into(),
        category: category.into(),
        language: language.into(),
        context: context.into(),
        body: body.into(),
        examples: examples.iter().map(|s| (*s).into()).collect(),
        counterexamples: counterexamples.iter().map(|s| (*s).into()).collect(),
    }
}

/// The shipped rules. Lexical, deliberately conservative, and every one carries
/// the examples it is certified against. Extend by editing the JSON that
/// `dblocks lm rules --out` writes.
fn builtin_rules() -> RuleSet {
    let categories = vec![
        Category {
            name: "error-swallowing".into(),
            weight: 1.0,
            description: "A failure is caught and discarded: the handler is empty or the error value is thrown away."
                .into(),
        },
        Category {
            name: "broad-catch".into(),
            weight: 0.5,
            description: "Every exception type is caught, so programming errors are handled like expected failures."
                .into(),
        },
        Category {
            name: "suppressed-diagnostics".into(),
            weight: 0.5,
            description: "A checker's warning is silenced wholesale instead of addressed or scoped.".into(),
        },
        Category {
            name: "hardcoded-secret".into(),
            weight: 1.0,
            description: "A credential is written into source rather than read from the environment.".into(),
        },
    ];

    let swallow = "error-swallowing";
    let broad = "broad-catch";
    let suppress = "suppressed-diagnostics";
    let secret = "hardcoded-secret";

    let rules = vec![
        // ---- error swallowing -------------------------------------------
        rule(
            "except-pass",
            swallow,
            "python",
            r"except\b[^:\n]*:\s*",
            r"pass\b",
            &[
                "try:\n    f()\nexcept:\n    pass\n",
                "except ValueError:\n    pass\n",
                "except (IOError, OSError) as e:\n        pass\n",
                "except Exception: pass\n",
            ],
            &[
                "except ValueError:\n    raise\n",
                "except Exception as e:\n    log.exception(e)\n    pass\n",
                "except:\n    passthrough()\n",
            ],
        ),
        rule(
            "except-ellipsis",
            swallow,
            "python",
            r"except\b[^:\n]*:\s*",
            r"\.\.\.",
            &["except KeyError:\n    ...\n", "except: ...\n"],
            &["except KeyError:\n    return ...\n"],
        ),
        rule(
            "empty-catch-block",
            swallow,
            "js/ts/java/c#/c++",
            r"catch\b[^{\n]*\{\s*",
            r"\}",
            &[
                "try { f(); } catch (e) {}",
                "} catch (err) { }",
                "catch (Exception e) {\n}",
                "} catch {\n    }\n",
            ],
            &[
                "catch (e) { throw e; }",
                "catch (e) { console.error(e); }",
                "catch (e) { /* intentionally ignored: best-effort cleanup */ }",
            ],
        ),
        rule(
            "catch-print-stack-trace-only",
            swallow,
            "java",
            r"catch\b[^{\n]*\{\s*\w+\.printStackTrace\(\)\s*;\s*",
            r"\}",
            &["catch (IOException e) {\n    e.printStackTrace();\n}"],
            &["catch (IOException e) {\n    e.printStackTrace();\n    throw new RuntimeException(e);\n}"],
        ),
        rule(
            "promise-catch-noop",
            swallow,
            "js/ts",
            r"\.catch\s*\(\s*\(?\s*\w*\s*\)?\s*=>\s*\{\s*",
            r"\}",
            &[
                "fetch(url).catch(() => {});",
                "p.catch(e => {})",
                "p.catch((err) => { })",
            ],
            &["p.catch(e => { report(e); })", "p.catch(() => retry())"],
        ),
        rule(
            "promise-catch-noop-function",
            swallow,
            "js",
            r"\.catch\s*\(\s*function\s*\([^)\n]*\)\s*\{\s*",
            r"\}",
            &["p.catch(function (e) {})", "p.catch(function(){ })"],
            &["p.catch(function (e) { log(e); })"],
        ),
        rule(
            "if-let-err-empty",
            swallow,
            "rust",
            r"if\s+let\s+Err\s*\(\s*_?\w*\s*\)\s*=[^{\n]*\{\s*",
            r"\}",
            &["if let Err(_) = fs::remove_file(p) {}", "if let Err(e) = tx.send(x) {\n}"],
            &["if let Err(e) = tx.send(x) { warn!(\"{e}\"); }"],
        ),
        rule(
            "match-err-arm-empty",
            swallow,
            "rust",
            r"Err\s*\(\s*_?\w*\s*\)\s*=>\s*",
            r"\{\s*\}",
            &["Err(_) => {}", "Err(e) => { }", "Err(_) => {},"],
            &["Err(e) => { return Err(e.into()); }", "Err(_) => { retry += 1; }"],
        ),
        rule(
            "match-err-arm-unit",
            swallow,
            "rust",
            r"Err\s*\(\s*_?\w*\s*\)\s*=>\s*",
            r"\(\)",
            &["Err(_) => (),", "Err(e) => ()"],
            &["Err(_) => (a, b),", "Err(e) => (e.into())"],
        ),
        rule(
            "go-empty-error-check",
            swallow,
            "go",
            r"if\s+err\s*!=\s*nil\s*\{\s*",
            r"\}",
            &["if err != nil {}", "if err != nil {\n}\n"],
            &["if err != nil {\n    return err\n}"],
        ),
        rule(
            "go-error-blank-identifier",
            swallow,
            "go",
            r",\s*",
            r"_\s*:?=\s*[\w.]+\(",
            &["n, _ := strconv.Atoi(s)", "data, _ = os.ReadFile(path)"],
            &["for i, _ := range xs {", "n, err := strconv.Atoi(s)", "_, err := f()"],
        ),
        // ---- broad catch --------------------------------------------------
        rule(
            "bare-except",
            broad,
            "python",
            r"except\s*",
            r":",
            &["except:\n    raise\n", "try:\n  f()\nexcept :\n  pass"],
            &["except ValueError:\n    raise\n", "except (A, B):\n    pass"],
        ),
        rule(
            "except-base-exception",
            broad,
            "python",
            r"except\s+",
            r"BaseException\b",
            &["except BaseException:\n    log()", "except BaseException as e:"],
            &["except Exception as e:", "except MyBaseExceptionSubclass:"],
        ),
        rule(
            "catch-throwable",
            broad,
            "java",
            r"catch\s*\(\s*",
            r"Throwable\b",
            &["catch (Throwable t) {", "catch(Throwable ignored) {}"],
            &["catch (IOException e) {", "catch (ThrowableSubtype t) {"],
        ),
        // ---- suppressed diagnostics --------------------------------------
        rule(
            "ts-ignore",
            suppress,
            "typescript",
            "",
            r"@ts-ignore\b",
            &["// @ts-ignore\nfoo.bar();", "/* @ts-ignore */"],
            &["// @ts-expect-error: upstream types are wrong\nfoo.bar();"],
        ),
        rule(
            "type-ignore-blanket",
            suppress,
            "python",
            r"#\s*type:\s*",
            r"ignore[ \t]*$",
            &["x = f()  # type: ignore\n", "y = g()  # type: ignore"],
            &["x = f()  # type: ignore[attr-defined]\n"],
        ),
        rule(
            "noqa-blanket",
            suppress,
            "python",
            r"#\s*",
            r"noqa[ \t]*$",
            &["import os  # noqa\n", "x = 1 # noqa"],
            &["import os  # noqa: F401\n", "# noqa is explained below\n"],
        ),
        rule(
            "eslint-disable-next-line-blanket",
            suppress,
            "js/ts",
            r"//\s*eslint-disable-next-line",
            r"[ \t]*\n",
            &["// eslint-disable-next-line\nconst x = 1;", "  // eslint-disable-next-line  \n"],
            &["// eslint-disable-next-line no-console\nconsole.log(x);"],
        ),
        rule(
            "eslint-disable-file-blanket",
            suppress,
            "js/ts",
            r"/\*\s*eslint-disable",
            r"\s*\*/",
            &["/* eslint-disable */\n", "/*eslint-disable*/"],
            &["/* eslint-disable no-console */\n"],
        ),
        rule(
            "allow-unused-must-use",
            suppress,
            "rust",
            r"#\s*!?\s*\[\s*allow\s*\(\s*",
            r"unused_must_use\b",
            &["#[allow(unused_must_use)]\nfn f() {}", "#![allow(unused_must_use)]"],
            &["#[allow(dead_code)]", "#[deny(unused_must_use)]"],
        ),
        // ---- hardcoded secrets -------------------------------------------
        rule(
            "password-literal",
            secret,
            "any",
            r#"[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]["']?\s*[:=]\s*["']"#,
            r#"[^"'\n$<{][^"'\n]*["']"#,
            &[
                "password = \"hunter2\"\n",
                "PASSWORD: 'correct horse'\n",
                "\"password\": \"s3cret\",",
            ],
            &[
                "password = os.environ[\"PASSWORD\"]",
                "password = \"\"",
                "password = \"${DB_PASSWORD}\"",
                "password = \"<your password here>\"",
                "pw = input(\"Password: \")",
            ],
        ),
        rule(
            "api-key-literal",
            secret,
            "any",
            r#"[Aa][Pp][Ii]_?[Kk][Ee][Yy]["']?\s*[:=]\s*["']"#,
            r#"[^"'\n$<{][^"'\n]*["']"#,
            &["api_key = \"sk-abc123\"", "API_KEY: 'x9'"],
            &["api_key = \"\"", "api_key = load_key()", "API_KEY = \"${API_KEY}\""],
        ),
        rule(
            "aws-access-key-id",
            secret,
            "any",
            "",
            r"AKIA[0-9A-Z]{16}\b",
            &["aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"],
            &["AKIA is the prefix", "AKIAIOSFODNN7EXAMPLE1234"],
        ),
        rule(
            "private-key-block",
            secret,
            "any",
            "",
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            &["-----BEGIN RSA PRIVATE KEY-----\nMIIE...", "-----BEGIN PRIVATE KEY-----"],
            &["-----BEGIN CERTIFICATE-----", "-----BEGIN PUBLIC KEY-----"],
        ),
    ];

    RuleSet { categories, rules }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(pattern: &str, text: &str) -> Option<(usize, usize)> {
        let p = Pattern::parse(pattern).unwrap();
        let tokens = text_tokens(text);
        (0..=tokens.len()).find_map(|at| p.match_at(&tokens, at).map(|(end, _)| (at, end)))
    }

    #[test]
    fn test_literals_classes_and_quantifiers() {
        assert_eq!(matches("abc", "xxabcxx"), Some((2, 5)));
        assert_eq!(matches("a.c", "abc a\nc"), Some((0, 3)));
        assert_eq!(matches("a.c", "a\nc"), None, "'.' must not cross a newline");
        assert_eq!(matches(r"\d+", "ab123c"), Some((2, 5)));
        assert_eq!(matches(r"[a-c]+", "xxcabz"), Some((2, 5)));
        assert_eq!(matches(r"[^)]*\)", "f(a, b) + 1"), Some((0, 7)));
        assert_eq!(matches(r"ab?c", "ac"), Some((0, 2)));
        assert_eq!(matches(r"ab?c", "abc"), Some((0, 3)));
        assert_eq!(matches(r"a{3}", "aaaa"), Some((0, 3)));
        assert_eq!(matches(r"a{2,3}b", "aaaab"), Some((1, 5)));
        assert_eq!(matches(r"a{2,}b", "ab"), None);
        assert_eq!(matches(r"[\w.]+\(", "x = foo.bar("), Some((4, 12)));
    }

    #[test]
    fn test_greedy_quantifiers_backtrack() {
        // `\s*` must give back the whitespace it ate when the rest of the
        // pattern needs it -- and `[^:\n]*` must stop before the colon.
        assert_eq!(matches(r"except\b[^:\n]*:\s*pass", "except ValueError:\n    pass"), Some((0, 27)));
        assert_eq!(matches(r"\s*$", "abc  \ndef"), Some((3, 5)));
        assert_eq!(matches(r"x[ \t]*$", "x   \n"), Some((0, 4)));
    }

    #[test]
    fn test_word_boundary_and_line_end() {
        assert_eq!(matches(r"\bpass\b", "passthrough pass"), Some((12, 16)));
        assert_eq!(matches(r"\bpass\b", "passthrough"), None);
        assert_eq!(matches(r"ignore$", "# type: ignore\nx"), Some((8, 14)));
        assert_eq!(matches(r"ignore$", "# type: ignore[x]\n"), None);
        assert_eq!(matches(r"end$", "the end"), Some((4, 7)), "$ matches at the end of the text");
    }

    #[test]
    fn test_unsupported_syntax_is_an_error_not_a_literal() {
        for bad in ["(a|b)", "a|b", "*a", "a{", "a{3,1}", "[abc", r"\q", "a\\", r"\b+", "a]"] {
            assert!(Pattern::parse(bad).is_err(), "{bad:?} should be rejected");
        }
        // ...while every escape the docs list parses.
        for good in [r"\(\)\[\]\{\}\*\+\?\.\|\\\^\$\/\-\#", r"[\]\[\-\\\s\w\d\n\t]", "[]a]", "[^]a]"] {
            assert!(Pattern::parse(good).is_ok(), "{good:?} should parse");
        }
    }

    #[test]
    fn test_specials_stop_every_match() {
        // A special token is not a byte: no atom may match it, including a
        // negated class and `.`, so a rule cannot reach across documents.
        let p = Pattern::parse(r"a[^z]b").unwrap();
        assert!(p.match_at(&[97, 120, 98], 0).is_some());
        assert!(p.match_at(&[97, 256, 98], 0).is_none());
        let any = Pattern::parse(r"a.b").unwrap();
        assert!(any.match_at(&[97, 258, 98], 0).is_none());
        // ...and a special is not a word character either.
        let boundary = Pattern::parse(r"\bab").unwrap();
        assert!(boundary.match_at(&[256, 97, 98], 1).is_some());
    }

    #[test]
    fn test_body_alone_is_labeled_and_first_span_wins() {
        let labeler = Labeler::builtin();
        let text = "try:\n    f()\nexcept:\n    pass\n";
        let (labels, spans) = labeler.label_with_spans(&text_tokens(text));
        let bytes = text.as_bytes();

        // `except:` fires bare-except (body ":") and except-pass (body "pass");
        // nothing in the context is labeled.
        let labeled: String = labels
            .iter()
            .zip(bytes)
            .filter(|(l, _)| **l != CLEAN)
            .map(|(_, b)| char::from(*b))
            .collect();
        assert_eq!(labeled, ":pass");
        assert_eq!(spans.len(), 2);

        let pass_at = text.find("pass").unwrap();
        let swallow = labeler.rule_set().category_index("error-swallowing").unwrap() as u8 + 1;
        let broad = labeler.rule_set().category_index("broad-catch").unwrap() as u8 + 1;
        assert!(labels[pass_at..pass_at + 4].iter().all(|l| *l == swallow));
        assert_eq!(labels[text.find(':').unwrap()], CLEAN, "the first colon belongs to `try:`");
        assert_eq!(labels[text.rfind(':').unwrap()], broad);
        assert_eq!(labeler.category_of(swallow).unwrap().name, "error-swallowing");
        assert_eq!(labeler.category_of(CLEAN), None);
    }

    #[test]
    fn test_overlapping_rules_are_both_reported_but_one_label_survives() {
        let labeler = Labeler::builtin();
        // Empty *and* catches Throwable: two findings, and the first span in
        // scan order owns the shared tokens.
        let text = "catch (Throwable t) {}";
        let (labels, spans) = labeler.label_with_spans(&text_tokens(text));
        let names = labeler.rule_names();
        let found: Vec<&str> = spans.iter().map(|s| names[s.rule]).collect();
        assert_eq!(found, vec!["empty-catch-block", "catch-throwable"]);
        assert_eq!(labels.iter().filter(|l| **l != CLEAN).count(), "}".len() + "Throwable".len());
    }

    #[test]
    fn test_optional_context_reports_each_body_once() {
        let labeler = Labeler::builtin();
        // `go-error-blank-identifier` has a context that starts with `,\s*`,
        // so it can anchor at several positions onto one body. One finding.
        let spans = labeler.scan(&text_tokens("n,    _ := strconv.Atoi(s)"));
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn test_builtin_rules_pass_their_own_examples() {
        // The examples on every rule are the rule's specification; the
        // certificate suite checks the same thing so a regression in the
        // matcher is a named failure.
        RuleSet::builtin().validate().unwrap();
        let labeler = Labeler::builtin();
        for rule in &labeler.rule_set().rules {
            assert!(!rule.examples.is_empty(), "rule '{}' has no examples", rule.name);
            assert!(!rule.counterexamples.is_empty(), "rule '{}' has no counterexamples", rule.name);
        }
    }

    #[test]
    fn test_rule_set_round_trips_through_json_and_rejects_bad_sets() {
        let set = RuleSet::builtin();
        let json = set.to_json().unwrap();
        assert_eq!(RuleSet::from_json(&json).unwrap(), set);

        let mut unknown = set.clone();
        unknown.rules[0].category = "no-such-category".into();
        assert!(unknown.validate().unwrap_err().to_string().contains("unknown category"));

        let mut empty_body = set.clone();
        empty_body.rules[0].body = r"\s*".into();
        assert!(empty_body.validate().unwrap_err().to_string().contains("empty string"));

        let mut failing_example = set.clone();
        failing_example.rules[0].examples.push("nothing to see here".into());
        assert!(failing_example.validate().unwrap_err().to_string().contains("own example"));

        let mut failing_counter = set.clone();
        failing_counter.rules[0].counterexamples.push("except:\n    pass".into());
        assert!(failing_counter.validate().unwrap_err().to_string().contains("counterexample"));

        let mut negative = set.clone();
        negative.categories[0].weight = -1.0;
        assert!(negative.validate().is_err());

        let mut duplicate = set;
        duplicate.categories.push(duplicate.categories[0].clone());
        assert!(duplicate.validate().unwrap_err().to_string().contains("twice"));
    }

    #[test]
    fn test_manifest_counts_and_weight_table() {
        let labeler = Labeler::builtin();
        let text = "except:\n    pass\ncatch (e) {}\n";
        let tokens = text_tokens(text);
        let (labels, spans) = labeler.label_with_spans(&tokens);
        let manifest = labeler.manifest(&labels, &spans);
        assert_eq!(manifest.tokens, tokens.len());
        assert_eq!(manifest.labeled_tokens, labels.iter().filter(|l| **l != CLEAN).count());
        let swallow = manifest.categories.iter().find(|c| c.name == "error-swallowing").unwrap();
        assert_eq!(swallow.spans, 2, "except-pass and empty-catch-block");
        assert_eq!(swallow.tokens, "pass".len() + "}".len());
        let table = manifest.weight_table();
        assert_eq!(table[0], 0.0, "clean tokens carry no weight");
        assert_eq!(table[1], labeler.categories()[0].weight);
        assert_eq!(labeler.weight_table(), table);

        let dir = std::env::temp_dir().join("dblocks-antipattern-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        manifest.write(&path).unwrap();
        assert_eq!(LabelManifest::read(&path).unwrap(), manifest);
        assert!(manifest.render().contains("error-swallowing"));
    }

    #[test]
    fn test_report_names_the_line_and_the_category() {
        let labeler = Labeler::builtin();
        let report = labeler.report("x = 1\ntry:\n    f()\nexcept:\n    pass\n");
        assert!(report.contains("line 5: [error-swallowing] except-pass -> \"pass\""), "{report}");
        assert!(report.contains("line 4: [broad-catch] bare-except"), "{report}");
    }
}
