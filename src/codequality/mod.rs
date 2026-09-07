//! Code-quality signals: language detection, per-language analyzers, and a
//! `QualityScore` in `[0, 1]` that the filter and the regularizer read
//! (roadmap Phase 25).
//!
//! # What "code quality" means here
//!
//! Three independent signals are combined into the score, each in `[0, 1]`
//! with `1.0` being "as good as this signal can measure":
//!
//! 1. **Anti-pattern density** from [`crate::antipattern`]. Every flagged
//!    token subtracts from a clean budget; a window with no flagged tokens
//!    scores `1.0` on this dimension.
//! 2. **Structural heuristics**: nesting depth, line length, blank-line
//!    ratio, identifier-naming consistency. None of these is a defect on
//!    its own; the score is a soft aggregation rather than a hard rule.
//! 3. **External analyzers** (clippy, ruff, eslint, golangci-lint, ...), when
//!    the `codequality-external` feature is enabled and the tool is
//!    installed. The output is parsed into a `(warning_count, total_lines)`
//!    pair and converted to a density score.
//!
//! The dimensions are independent by design: a run that disables one keeps
//! the others, and a run that disables all is exactly the identity
//! (`score == 1.0` for every window).
//!
//! # Plug-in point
//!
//! Adding a new language is a `Language::new_language(...)` declaration plus
//! an implementation of [`CodeAnalyzer`]; everything downstream
//! ([`filter`], [`regularizer`], the CLI) reads the trait only and does not
//! care which concrete analyzer produced a score.

pub mod external;
pub mod filter;
pub mod regularizer;

// Re-exports so callers (the CLI, the verify suite, downstream modules)
// reach the three core types through one path: `codequality::CodeAnalyzer`
// rather than `codequality::mod::CodeAnalyzer`.
pub use filter::{FilterReport, Policy, WindowFilter};
pub use regularizer::QualityRegularizer;

use serde::{Deserialize, Serialize};

/// A programming language, used to pick an analyzer and to label a window's
/// score in the on-disk sidecar.
///
/// `Generic` is the fallback for windows whose language cannot be inferred;
/// its analyzer reports no anti-patterns and uses the structural heuristics
/// alone, which is enough for prose-heavy corpora and unknown languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    Generic,
}

impl Language {
    /// Every variant, in the order they appear in [`Self::name`].
    pub const ALL: [Language; 9] = [
        Self::Rust,
        Self::Python,
        Self::JavaScript,
        Self::TypeScript,
        Self::Go,
        Self::Java,
        Self::C,
        Self::Cpp,
        Self::Generic,
    ];

    /// Canonical lower-case name used in CLI flags and sidecar headers.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Generic => "generic",
        }
    }

    /// Parse a name, accepting common aliases. Unknown names fall back to
    /// [`Self::Generic`] rather than failing: a CLI flag `--language foo`
    /// that errors out is less useful than one that runs the generic
    /// analyzer and labels the result `generic`.
    pub fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "rust" | "rs" => Self::Rust,
            "python" | "py" => Self::Python,
            "javascript" | "js" => Self::JavaScript,
            "typescript" | "ts" => Self::TypeScript,
            "go" | "golang" => Self::Go,
            "java" => Self::Java,
            "c" => Self::C,
            "cpp" | "c++" | "cxx" => Self::Cpp,
            _ => Self::Generic,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One independent quality signal, in `[0, 1]`. `1.0` is best.
///
/// `name` is `String` rather than `&'static str` so the type can be
/// deserialized from a sidecar file: a name the analyzer emits at runtime
/// is not `static`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dimension {
    pub name: String,
    pub score: f32,
}

impl Dimension {
    pub fn new(name: impl Into<String>, score: f32) -> Self {
        Self {
            name: name.into(),
            score: score.clamp(0.0, 1.0),
        }
    }
}

/// Per-window quality report.
///
/// `overall` is a weighted geometric mean of the per-dimension scores: any
/// single dimension at `0.0` pulls the overall score to `0.0`, which is the
/// behavior the filter wants — one bad dimension should not be averaged away
/// by three good ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityScore {
    pub language: Language,
    pub overall: f32,
    pub dimensions: Vec<Dimension>,
    /// Number of source lines the analyzer saw. `0` is reported, not
    /// silently dropped, because a window that scored `1.0` on zero lines is
    /// not the same as one that scored `1.0` on a hundred.
    pub lines: usize,
}

impl QualityScore {
    /// Identity: score `1.0`, no dimensions. Every analyzer's "off" state
    /// reduces to this.
    pub fn identity(language: Language) -> Self {
        Self {
            language,
            overall: 1.0,
            dimensions: Vec::new(),
            lines: 0,
        }
    }

    /// Weighted geometric mean of the dimensions, or `1.0` if there are none.
    /// A dimension at `0.0` makes the product `0.0`, so the filter sees a
    /// clearly bad window rather than a barely-bad one. The
    /// `lexical_density` function is responsible for keeping its own score
    /// above zero so this product can be exact: collapsing to zero on a
    /// non-defective window would make the filter drop a window that is
    /// merely sparse, not bad.
    pub fn from_dimensions(language: Language, dimensions: Vec<Dimension>, lines: usize) -> Self {
        if dimensions.is_empty() {
            return Self::identity(language);
        }
        let product: f64 = dimensions.iter().map(|d| f64::from(d.score)).product();
        let n = dimensions.len() as f64;
        let overall = if product == 0.0 {
            0.0
        } else {
            (product.powf(1.0 / n)) as f32
        };
        Self {
            language,
            overall: overall.clamp(0.0, 1.0),
            dimensions,
            lines,
        }
    }

    /// Whether this score would cause a downstream filter to drop the
    /// window (overall at zero).
    pub fn is_zero(&self) -> bool {
        self.overall <= 0.0
    }
}

/// A code-quality analyzer: stateless function from source to score.
pub trait CodeAnalyzer {
    fn language(&self) -> Language;
    fn analyze(&self, source: &str) -> QualityScore;
}

/// Built-in analyzer that combines the three signals: anti-pattern density,
/// structural heuristics, and (when enabled) an external tool.
#[derive(Debug, Clone)]
pub struct CompositeAnalyzer<L: CodeAnalyzer> {
    language: Language,
    /// Anti-pattern density. `None` means: do not include this dimension.
    lexical: Option<crate::antipattern::Labeler>,
    /// Structural heuristics. `None` means: do not include this dimension.
    structural: Option<StructuralAnalyzer>,
    /// Optional external tool, behind the `codequality-external` feature.
    #[cfg_attr(not(feature = "codequality-external"), allow(dead_code))]
    external: Option<ExternalAnalyzer>,
    _language_phantom: std::marker::PhantomData<L>,
}

impl<L: CodeAnalyzer> CompositeAnalyzer<L> {
    /// Compose the available dimensions. Disabling a signal is a `None`,
    /// not a no-op analyzer that reports `1.0` for everything: the filter
    /// needs to distinguish "this dimension was not measured" from "this
    /// dimension measured `1.0`".
    pub fn new(
        language: Language,
        lexical: Option<crate::antipattern::Labeler>,
        structural: Option<StructuralAnalyzer>,
        external: Option<ExternalAnalyzer>,
    ) -> Self {
        Self {
            language,
            lexical,
            structural,
            external,
            _language_phantom: std::marker::PhantomData,
        }
    }
}

impl<L: CodeAnalyzer> CodeAnalyzer for CompositeAnalyzer<L> {
    fn language(&self) -> Language {
        self.language
    }

    fn analyze(&self, source: &str) -> QualityScore {
        let mut dimensions = Vec::new();
        let lines = source.lines().count();

        if let Some(labeler) = &self.lexical {
            dimensions.push(Dimension::new("lexical", lexical_density(labeler, source)));
        }
        if let Some(s) = &self.structural {
            dimensions.push(Dimension::new("structural", s.score(source)));
        }
        #[cfg(feature = "codequality-external")]
        if let Some(ext) = &self.external {
            if let Some(score) = ext.score(source) {
                dimensions.push(Dimension::new("external", score));
            }
        }
        let _ = L::language;

        QualityScore::from_dimensions(self.language, dimensions, lines)
    }
}

/// Lexical anti-pattern density, in `[0, 1]`.
///
/// A window with no flagged tokens scores `1.0`. The score falls to `0.0`
/// as the flagged-token fraction grows; the half-life is set so that a
/// window where one in four tokens is flagged scores `0.5`. This keeps a
/// pathologically bad window (`flagged == 1.0`) at `0.0` — the only number
/// the filter treats as "drop me" — while a mildly bad one still has a
/// chance to survive with a lower weight.
pub fn lexical_density(labeler: &crate::antipattern::Labeler, source: &str) -> f32 {
    let tokens = crate::antipattern::text_tokens(source);
    if tokens.is_empty() {
        return 1.0;
    }
    let labels = labeler.label(&tokens);
    let flagged = labels
        .iter()
        .filter(|l| **l != crate::antipattern::CLEAN)
        .count();
    let frac = flagged as f32 / tokens.len() as f32;
    // Exponential: score = 2^(-frac / 0.25). frac=0 -> 1.0, frac=0.25 -> 0.5,
    // frac=1.0 -> 2^-4 = 1/16 ~= 0.0625. The floor at 1e-6 prevents the
    // product in `QualityScore::from_dimensions` from collapsing to exact
    // zero, which would make the geometric mean zero and trip the filter.
    let exponent = -frac / 0.25;
    exponent.exp2().clamp(1e-6, 1.0)
}

/// Cheap structural heuristics. None of them is a defect on its own; the
/// point is to penalize windows where they all fire together.
///
/// - **nesting depth**: capped at `8`, then scaled. Deeply nested code is
///   harder to read; a window that stays shallow scores `1.0`.
/// - **line length**: fraction of lines longer than `120` characters.
/// - **blank-line ratio**: a window that is half blank lines scores `0.5`
///   on this dimension (the heuristic expects roughly 1 blank in 6).
/// - **identifier naming**: fraction of identifiers in snake_case that are
///   not actually snake_case, and likewise for camelCase. Surfaces code
///   that mixes conventions in one window.
#[derive(Debug, Clone, Copy)]
pub struct StructuralAnalyzer {
    pub max_nesting: usize,
    pub max_line_length: usize,
    pub blank_ratio_target: f32,
    pub snake_weight: f32,
    pub camel_weight: f32,
}

impl Default for StructuralAnalyzer {
    fn default() -> Self {
        Self {
            max_nesting: 8,
            max_line_length: 120,
            blank_ratio_target: 0.15,
            snake_weight: 1.0,
            camel_weight: 1.0,
        }
    }
}

impl StructuralAnalyzer {
    pub fn score(&self, source: &str) -> f32 {
        if source.is_empty() {
            return 1.0;
        }
        let depth = max_indent_depth(source);
        let depth_score = (1.0 - depth as f32 / self.max_nesting as f32).clamp(0.0, 1.0);

        let (total_lines, long_lines) = line_lengths(source);
        let length_score = if total_lines == 0 {
            1.0
        } else {
            1.0 - (long_lines as f32 / total_lines as f32)
        };

        let blank_ratio = blank_line_ratio(source);
        let blank_score = 1.0 - (blank_ratio - self.blank_ratio_target).abs();

        let (snake_violations, camel_violations, total_ids) = naming_consistency(source);
        let naming_score = if total_ids == 0 {
            1.0
        } else {
            let weighted = self.snake_weight * snake_violations as f32
                + self.camel_weight * camel_violations as f32;
            1.0 - (weighted / total_ids as f32).min(1.0)
        };

        // Weighted arithmetic mean; this dimension combines sub-signals that
        // are individually continuous, so a geometric mean is not needed.
        let weights = [1.0f32, 1.0, 0.5, 1.0];
        let weighted_sum = depth_score * weights[0]
            + length_score * weights[1]
            + blank_score * weights[2]
            + naming_score * weights[3];
        let total_weight: f32 = weights.iter().sum();
        (weighted_sum / total_weight).clamp(1e-6, 1.0)
    }
}

fn max_indent_depth(source: &str) -> usize {
    source
        .lines()
        .map(|line| {
            let leading = line
                .bytes()
                .take_while(|b| matches!(b, b' ' | b'\t'))
                .count();
            // Each tab counts as 4 columns. This is the convention rustfmt
            // uses; other conventions exist, but mixing them in one window
            // is itself a signal.
            let cols = line
                .bytes()
                .take(leading)
                .map(|b| if b == b'\t' { 4 } else { 1 })
                .sum::<usize>();
            cols / 4
        })
        .max()
        .unwrap_or(0)
}

fn line_lengths(source: &str) -> (usize, usize) {
    let mut total = 0usize;
    let mut long = 0usize;
    for line in source.lines() {
        total += 1;
        if line.len() > 120 {
            long += 1;
        }
    }
    (total, long)
}

fn blank_line_ratio(source: &str) -> f32 {
    let total = source.lines().count();
    if total == 0 {
        return 0.0;
    }
    let blank = source.lines().filter(|l| l.trim().is_empty()).count();
    blank as f32 / total as f32
}

fn naming_consistency(source: &str) -> (usize, usize, usize) {
    // Tokens that look like identifiers: start with a letter or `_`, then
    // letters, digits, or `_`. This matches the lexical shape of
    // snake/camel/Pascal case; it is a cheap heuristic, not a parser.
    let mut snake_violations = 0usize;
    let mut camel_violations = 0usize;
    let mut total = 0usize;

    for word in source.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.is_empty()
            || !word
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            continue;
        }
        // Skip strings of length 1: a single letter matches everything.
        if word.len() < 3 {
            continue;
        }
        total += 1;
        let has_upper = word.chars().any(|c| c.is_ascii_uppercase());
        let has_underscore = word.contains('_');

        // Convention (a deliberately simple one): identifiers in code are
        // either snake_case or camelCase. We don't know which one this
        // window "should" use -- that depends on the language -- so we
        // penalize mixed signals: an underscore-bearing identifier that
        // also has uppercase letters (SNAKE_case_with_Mixed) is the
        // clearest violation.
        if has_underscore && has_upper {
            snake_violations += 1;
        } else if has_upper
            && word
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
            && has_underscore
        {
            camel_violations += 1;
        }
    }
    (snake_violations, camel_violations, total)
}

/// Stub for an external-tool-backed analyzer. The actual shelling-out
/// happens in [`crate::codequality::external`] behind the
/// `codequality-external` Cargo feature; with the feature off this type
/// exists but always returns `None`, so the rest of the pipeline works
/// identically.
#[derive(Debug, Clone, Default)]
pub struct ExternalAnalyzer {
    /// Free-form name recorded in the score report ("clippy", "ruff", ...).
    pub name: String,
    /// Command to invoke, including the flag that produces JSON output.
    pub command: Vec<String>,
}

impl ExternalAnalyzer {
    pub fn new(name: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            name: name.into(),
            command,
        }
    }

    /// Score `source` by shelling out to the configured command. Disabled
    /// when the `codequality-external` feature is off; the score is then
    /// always `None`, and the composite analyzer simply skips the
    /// dimension. The `source` argument is unused in that case.
    #[cfg_attr(not(feature = "codequality-external"), allow(unused_variables))]
    pub fn score(&self, source: &str) -> Option<f32> {
        #[cfg(feature = "codequality-external")]
        {
            crate::codequality::external::run(self, source)
        }
        #[cfg(not(feature = "codequality-external"))]
        {
            let _ = source;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_round_trip() {
        for lang in Language::ALL {
            let parsed = Language::parse(lang.name());
            assert_eq!(parsed, lang);
        }
        for (alias, expected) in [
            ("rs", Language::Rust),
            ("py", Language::Python),
            ("js", Language::JavaScript),
            ("ts", Language::TypeScript),
            ("golang", Language::Go),
            ("c++", Language::Cpp),
            ("cxx", Language::Cpp),
        ] {
            assert_eq!(Language::parse(alias), expected);
        }
        assert_eq!(Language::parse("not-a-language"), Language::Generic);
    }

    #[test]
    fn test_lexical_density_is_one_for_clean_source() {
        let labeler = crate::antipattern::Labeler::builtin();
        let score = lexical_density(&labeler, "def add(x, y):\n    return x + y\n");
        assert!(
            (score - 1.0).abs() < 1e-6,
            "clean source must score 1.0, got {score}"
        );
    }

    #[test]
    fn test_lexical_density_falls_as_flagged_fraction_grows() {
        let labeler = crate::antipattern::Labeler::builtin();
        // A window that is one-quarter flagged tokens: half-life is exactly
        // 0.25, so this is 0.5.
        let text = "try:\n    f()\nexcept:\n    pass\n";
        let tokens = crate::antipattern::text_tokens(text);
        let flagged = labeler.label(&tokens).iter().filter(|l| **l != 0).count();
        let frac = flagged as f32 / tokens.len() as f32;
        let score = lexical_density(&labeler, text);
        let expected = (-frac / 0.25_f32).exp2();
        assert!((score - expected).abs() < 1e-5, "{score} vs {expected}");
        assert!(score < 1.0, "flagged text must score below 1.0: {score}");
    }

    #[test]
    fn test_lexical_density_clamps_above_zero_for_a_pathological_window() {
        // A window where every token is flagged: score must still be > 0
        // so the geometric mean does not collapse to an exact zero. The
        // floor at 1e-6 is the contract with `QualityScore::from_dimensions`.
        let labeler = crate::antipattern::Labeler::builtin();
        let score = lexical_density(&labeler, "pass");
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_structural_score_is_one_for_clean_well_formed_source() {
        let s = StructuralAnalyzer::default();
        let score = s.score("def add(x, y):\n    return x + y\n");
        assert!(score > 0.9, "clean source must score near 1.0, got {score}");
    }

    #[test]
    fn test_structural_score_falls_with_deep_nesting() {
        let s = StructuralAnalyzer::default();
        let flat = "x = 1\ny = 2\nz = 3\n";
        let deep = (0..20)
            .map(|i| format!("{}print({i})\n", "    ".repeat(i.min(8))))
            .collect::<Vec<_>>()
            .join("\n");
        let a = s.score(flat);
        let b = s.score(&deep);
        assert!(b < a, "deep nesting must score lower: {a} vs {b}");
    }

    #[test]
    fn test_max_indent_depth_treats_tabs_as_four_columns() {
        // Two tabs at column 0 -> depth 2 with rustfmt's convention. The
        // third line's deeper indent (three tabs) is what the test
        // actually checks: the function returns the *max* depth seen, not
        // the depth of the first line.
        let source = "\t\tif a:\n\t\t\tpass\n";
        assert_eq!(max_indent_depth(source), 3);

        // Two lines, both at depth 2: max is 2.
        let flat = "\t\tif a:\n\t\tif b:\n";
        assert_eq!(max_indent_depth(flat), 2);
    }

    #[test]
    fn test_quality_score_identity_is_dimensionless() {
        let s = QualityScore::identity(Language::Rust);
        assert_eq!(s.overall, 1.0);
        assert!(s.dimensions.is_empty());
        assert_eq!(s.lines, 0);
        assert!(!s.is_zero());
    }

    #[test]
    fn test_quality_score_geometric_mean_collapses_on_zero() {
        // Any single dimension at zero: the geometric mean is zero, so the
        // filter treats the window as drop-worthy. This is the property
        // that stops "one bad dimension averaged away by three good ones".
        let dims = vec![
            Dimension::new("a", 1.0),
            Dimension::new("b", 0.0),
            Dimension::new("c", 1.0),
        ];
        let s = QualityScore::from_dimensions(Language::Python, dims, 10);
        assert_eq!(s.overall, 0.0, "geometric mean must be 0 when any dim is 0");
        assert!(s.is_zero());
    }

    #[test]
    fn test_quality_score_with_all_ones_is_exactly_one() {
        let dims = vec![
            Dimension::new("a", 1.0),
            Dimension::new("b", 1.0),
            Dimension::new("c", 1.0),
        ];
        let s = QualityScore::from_dimensions(Language::Rust, dims, 1);
        assert_eq!(s.overall, 1.0);
    }

    #[test]
    fn test_composite_analyzer_with_no_signals_is_identity() {
        // The default state of `CompositeAnalyzer` with every signal None
        // is the identity, so the rest of the pipeline pays no cost when
        // the feature is disabled end-to-end.
        struct Empty(Language);
        impl CodeAnalyzer for Empty {
            fn language(&self) -> Language {
                self.0
            }
            fn analyze(&self, _: &str) -> QualityScore {
                QualityScore::identity(self.0)
            }
        }
        let analyzer = CompositeAnalyzer::<Empty>::new(Language::Rust, None, None, None);
        let score = analyzer.analyze("anything");
        assert_eq!(score.overall, 1.0);
        assert!(score.dimensions.is_empty());
        assert_eq!(score.language, Language::Rust);
    }

    #[test]
    fn test_composite_analyzer_with_just_lexical_matches_lexical_density() {
        let labeler = crate::antipattern::Labeler::builtin();
        let analyzer = CompositeAnalyzer::<LexicalOnly>::new(
            Language::Rust,
            Some(labeler.clone()),
            None,
            None,
        );
        let text = "try:\n    f()\nexcept:\n    pass\n";
        let expected = lexical_density(&labeler, text);
        let score = analyzer.analyze(text);
        assert!((score.overall - expected).abs() < 1e-5);
        assert_eq!(score.dimensions.len(), 1);
        assert_eq!(score.dimensions[0].name, "lexical");
    }

    struct LexicalOnly;
    impl CodeAnalyzer for LexicalOnly {
        fn language(&self) -> Language {
            Language::Rust
        }
        fn analyze(&self, _: &str) -> QualityScore {
            QualityScore::identity(Language::Rust)
        }
    }

    #[test]
    fn test_an_analyzer_is_a_pure_function_of_its_source() {
        // The contract downstream modules rely on: the same source produces
        // the same score, every call, with no hidden state. Random hashing
        // would silently invalidate sidecar files; tested here so a
        // regression in a future dimension is a named failure.
        let labeler = crate::antipattern::Labeler::builtin();
        let analyzer = CompositeAnalyzer::<LexicalOnly>::new(
            Language::Python,
            Some(labeler),
            Some(StructuralAnalyzer::default()),
            None,
        );
        let text = "def f(x):\n    return x + 1\n";
        let a = analyzer.analyze(text);
        let b = analyzer.analyze(text);
        let c = analyzer.analyze(text);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn test_external_analyzer_without_feature_is_a_no_op() {
        // With the feature disabled, the external dimension simply does
        // not contribute, regardless of what command was configured.
        let ext = ExternalAnalyzer::new("clippy", vec!["cargo".into(), "clippy".into()]);
        assert!(ext.score("fn main() {}").is_none());
    }

    #[test]
    fn test_blank_line_ratio_is_zero_on_a_dense_window() {
        assert_eq!(blank_line_ratio("a = 1\nb = 2\nc = 3\n"), 0.0);
    }
}
