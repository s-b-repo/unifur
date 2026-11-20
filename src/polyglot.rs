//! Multi-language cheat coverage: lint suppression, skipped and weakened
//! tests, stubs, type-checker bypasses, unsafe escapes and gate/config
//! weakening, for the languages and linters [`crate::cheat`] does not read and
//! for the gaps in the ones it does.
//!
//! # Where this sits
//!
//! [`crate::cheat::scan_snippet`] reads six language families with
//! hand-written passes. A coding specialist's corpus carries dozens of
//! languages, each with its own linters and each linter with its own
//! suppression spelling: `//nolint`, `# rubocop:disable`, `{-# OPTIONS_GHC -w
//! #-}`, `<NoWarn>`, `continue-on-error: true`. This module is the catalog for
//! that long tail. [`rules`] returns a language's rules as plain data,
//! [`scan`] applies them, and every [`Hit`] carries a [`PolyKind`] that maps
//! onto a [`crate::cheat::CheatClass`] (see "Integration" below).
//!
//! What cheat.rs already detects on the snippet path is *not* repeated:
//! Rust `#[allow]`/`#[expect]`/`#[cfg_attr(.., allow)]`/`#[ignore]`/`todo!()`,
//! unsafe without `SAFETY`; Python `noqa`/`type: ignore`/`pyright: ignore`/
//! `pylint: disable`, unconditional `pytest`/`unittest` skips, bare `except`,
//! `raise NotImplementedError`, `pass` bodies; JS/TS `eslint-disable*`,
//! `@ts-ignore`/`@ts-nocheck`/`@ts-expect-error`, `.skip`/`xit`; C-family
//! `NOLINT*` and diagnostic `#pragma`s (which also covers C#'s `#pragma
//! warning disable`, since C# scans as the C++ family). Rules that repeat a
//! *gate-side* check — one the snippet path never runs, such as `autotests =
//! false` in a manifest or a `clippy.toml` threshold of 1000 — are kept,
//! because a trace that *is* a manifest is otherwise unread, and each carries
//! [`Rule::overlap`] naming the cheat.rs function that also sees it, so a
//! caller running both over a candidate tree can drop the duplicates.
//!
//! # Directive position
//!
//! A suppression is a comment the tool *acts on*, and almost every tool acts
//! only when the directive opens the comment: flake8 reads `#\s*noqa`, ESLint
//! wants `eslint-` as the comment's first word, golangci-lint `//nolint`.
//! [`lex`] splits each line into a code view, a text view (comments removed,
//! string contents kept) and comment segments, with a per-language lexer that
//! knows line and nested block comments, quoted/char/raw/triple-quoted
//! strings, Rust `r#"…"#`, C++ `R"d(…)d"`, C# `@"…"`, Lua long brackets,
//! Perl/Ruby POD blocks, shell/Ruby/Perl/PHP/HCL heredocs and PowerShell
//! here-strings. A [`Site::Directive`] rule matches only at the start of a
//! comment (after `*`, `/`, `!`, `-` decoration), or at the start of a later
//! segment introduced by the same line marker (`# a  # noqa`). So `# see the
//! noqa docs` and `"// eslint-disable"` in a string are not suppressions,
//! while tools that genuinely match anywhere in a comment (clang-tidy, PMD's
//! `NOPMD`, gosec's `#nosec`, Sonar's `NOSONAR`) use [`Site::Comment`].
//!
//! # Justified suppressions
//!
//! Same policy as cheat.rs: a waivable rule ([`Waiver::Reason`]) is downgraded
//! to [`Severity::Note`] when a comment on the same line, or on a comment-only
//! line directly above, contains `reason` or `audit-allow:` *followed by
//! words*. [`Waiver::Native`] also accepts the tool's own justification slot
//! when it holds at least two words of three or more letters: ESLint/RuboCop/
//! phpcs `-- why`, Biome `: why`, `//nolint:x // why`, PHPStan `(why)`, C# and
//! PowerShell `Justification = "why"`, checkov `:why`. A native slot is easier
//! to fill with filler than the `reason` policy; that is a stated weakness,
//! not a hidden one. Skipped tests, stubs and whole-checker switches are
//! [`Waiver::Never`]: a reason does not make a disabled gate a working one.
//!
//! # Regression corpus, not a detection rate
//!
//! [`regression_cases`] holds hand-written cases — at least two positives,
//! one justified-suppression note and one clean negative per language, plus
//! negatives that only *mention* a directive in a string, docstring, heredoc
//! or prose comment — and [`regression_rates`] scores them. The numbers say
//! the catalog still classifies every listed case the way it did when the
//! case was written. They are **not** a recall or false-positive rate on real
//! code: the cases were written by the same hand as the rules, and
//! [`Rates::rules_fired`] shows how much of the catalog they even exercise.
//!
//! # What this does not claim
//!
//! Matching is line-oriented. A directive assembled at runtime, split across
//! a macro, or spelled through an alias is not seen; a YAML block scalar is
//! lexed as plain lines; Swift raw strings `#"…"#`, OCaml `{|…|}`, Perl
//! `q{…}`/`qq{…}`, Elixir sigils and string interpolation are lexed as code.
//! Multi-line config values (a `Checks: >` block, a JSON array spread over
//! lines) are read one line at a time. Unknown made-up `cfg` names, feature
//! flags and build tags cannot be known to be false. Like cheat.rs, this
//! raises the cost of cheating; it does not make it impossible.
//!
//! # Integration into `cheat::scan_snippet`
//!
//! Route by [`Lang::snippet_family`] (the existing pass) and run [`scan`] on
//! the same text with the precise [`Lang`]. Map each [`Hit`] to a
//! `Finding` with the hit's own severity (`Finding::with_severity`), class by
//! kind: `LintSuppression`/`TypeCheckBypass`/`UnsafeEscape`/`ConfigWeakening`
//! → `DataSuppression`, `TestSkip`/`TestWeakening` → `DataEvalGaming`,
//! `Stub` → `DataStub`. Evidence is the trimmed source line, as in cheat.rs,
//! so a hit on a line cheat.rs also flagged under the same class collapses in
//! the report's fingerprint de-duplication.

use crate::cheat::{Severity, SnippetLanguage};
use serde::{Serialize, Serializer};

// ---------------------------------------------------------------------------
// Languages
// ---------------------------------------------------------------------------

/// A language, or a linter/build configuration format treated as one.
///
/// Configuration formats are languages here because the cheat lives in them:
/// `continue-on-error: true` is a sentence in GitHub Actions, not in YAML.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lang {
    Rust,
    Python,
    C,
    Cpp,
    CSharp,
    Java,
    Kotlin,
    Scala,
    Go,
    JavaScript,
    TypeScript,
    Ruby,
    Php,
    Swift,
    ObjectiveC,
    Dart,
    Lua,
    Perl,
    R,
    Julia,
    Haskell,
    OCaml,
    Elixir,
    Erlang,
    Clojure,
    Zig,
    Nim,
    Solidity,
    /// sh, bash, zsh, ksh, dash; also `.shellcheckrc`.
    Shell,
    /// Scripts, modules and PSScriptAnalyzer settings files.
    PowerShell,
    /// Any dialect; also `.sqlfluff`.
    Sql,
    /// Terraform / HCL; also `.tflint.hcl`.
    Terraform,
    /// Dockerfile / Containerfile; also `.hadolint.yaml`.
    Dockerfile,
    /// Makefile, `*.mk`, justfile.
    Makefile,
    /// Generic YAML (Kubernetes, Ansible, ...).
    Yaml,
    /// CI pipelines: GitHub Actions, GitLab CI, Azure, CircleCI, Travis,
    /// Bitbucket, Jenkinsfile, pre-commit.
    CiConfig,
    /// `Cargo.toml`, `.cargo/config.toml`.
    CargoManifest,
    /// `clippy.toml`, `.clippy.toml`.
    ClippyConfig,
    /// `pyproject.toml`, `setup.cfg`, `tox.ini`, `.flake8`, `mypy.ini`,
    /// `.pylintrc`, `ruff.toml`, `pytest.ini`, `.coveragerc`,
    /// `pyrightconfig.json`.
    PythonConfig,
    /// `.eslintrc*`, `eslint.config.*`, `biome.json`, `.jshintrc`,
    /// `.solhint.json`, `.oxlintrc.json`.
    JsLintConfig,
    /// `tsconfig*.json`, `jsconfig.json`.
    TsConfig,
    /// `package.json` and test-runner configs (`jest.config.*`,
    /// `vitest.config.*`, `.mocharc.*`).
    NodePackage,
    /// `.golangci.{yml,yaml,toml,json}`.
    GolangciConfig,
    /// `*.csproj`, `*.vbproj`, `*.fsproj`, `*.props`, `*.targets`, `*.ruleset`.
    MsBuild,
    /// `.editorconfig`, `.globalconfig`.
    EditorConfig,
    /// `build.gradle(.kts)`, `settings.gradle(.kts)`.
    Gradle,
    /// `pom.xml` and Checkstyle/PMD/SpotBugs XML configs.
    Maven,
    /// `.rubocop.yml`, `.rubocop_todo.yml`, `.standard.yml`.
    RubocopConfig,
    /// `phpstan.neon*`, `psalm.xml*`, `phpcs.xml*`, `phpunit.xml*`.
    PhpLintConfig,
    /// `.swiftlint.yml`.
    SwiftlintConfig,
    /// `analysis_options.yaml`.
    DartAnalysis,
    /// `CMakeLists.txt`, `*.cmake`.
    CMake,
    /// `.clang-tidy`.
    ClangTidyConfig,
}

impl Serialize for Lang {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl Lang {
    /// Every language, in declaration order.
    pub const ALL: &'static [Lang] = &[
        Lang::Rust,
        Lang::Python,
        Lang::C,
        Lang::Cpp,
        Lang::CSharp,
        Lang::Java,
        Lang::Kotlin,
        Lang::Scala,
        Lang::Go,
        Lang::JavaScript,
        Lang::TypeScript,
        Lang::Ruby,
        Lang::Php,
        Lang::Swift,
        Lang::ObjectiveC,
        Lang::Dart,
        Lang::Lua,
        Lang::Perl,
        Lang::R,
        Lang::Julia,
        Lang::Haskell,
        Lang::OCaml,
        Lang::Elixir,
        Lang::Erlang,
        Lang::Clojure,
        Lang::Zig,
        Lang::Nim,
        Lang::Solidity,
        Lang::Shell,
        Lang::PowerShell,
        Lang::Sql,
        Lang::Terraform,
        Lang::Dockerfile,
        Lang::Makefile,
        Lang::Yaml,
        Lang::CiConfig,
        Lang::CargoManifest,
        Lang::ClippyConfig,
        Lang::PythonConfig,
        Lang::JsLintConfig,
        Lang::TsConfig,
        Lang::NodePackage,
        Lang::GolangciConfig,
        Lang::MsBuild,
        Lang::EditorConfig,
        Lang::Gradle,
        Lang::Maven,
        Lang::RubocopConfig,
        Lang::PhpLintConfig,
        Lang::SwiftlintConfig,
        Lang::DartAnalysis,
        Lang::CMake,
        Lang::ClangTidyConfig,
    ];

    /// Stable lowercase name; [`Lang::from_name`] accepts it back.
    pub fn as_str(self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::Python => "python",
            Lang::C => "c",
            Lang::Cpp => "cpp",
            Lang::CSharp => "csharp",
            Lang::Java => "java",
            Lang::Kotlin => "kotlin",
            Lang::Scala => "scala",
            Lang::Go => "go",
            Lang::JavaScript => "javascript",
            Lang::TypeScript => "typescript",
            Lang::Ruby => "ruby",
            Lang::Php => "php",
            Lang::Swift => "swift",
            Lang::ObjectiveC => "objective-c",
            Lang::Dart => "dart",
            Lang::Lua => "lua",
            Lang::Perl => "perl",
            Lang::R => "r",
            Lang::Julia => "julia",
            Lang::Haskell => "haskell",
            Lang::OCaml => "ocaml",
            Lang::Elixir => "elixir",
            Lang::Erlang => "erlang",
            Lang::Clojure => "clojure",
            Lang::Zig => "zig",
            Lang::Nim => "nim",
            Lang::Solidity => "solidity",
            Lang::Shell => "shell",
            Lang::PowerShell => "powershell",
            Lang::Sql => "sql",
            Lang::Terraform => "terraform",
            Lang::Dockerfile => "dockerfile",
            Lang::Makefile => "makefile",
            Lang::Yaml => "yaml",
            Lang::CiConfig => "ci-config",
            Lang::CargoManifest => "cargo-manifest",
            Lang::ClippyConfig => "clippy-config",
            Lang::PythonConfig => "python-config",
            Lang::JsLintConfig => "js-lint-config",
            Lang::TsConfig => "tsconfig",
            Lang::NodePackage => "node-package",
            Lang::GolangciConfig => "golangci-config",
            Lang::MsBuild => "msbuild",
            Lang::EditorConfig => "editorconfig",
            Lang::Gradle => "gradle",
            Lang::Maven => "maven",
            Lang::RubocopConfig => "rubocop-config",
            Lang::PhpLintConfig => "php-lint-config",
            Lang::SwiftlintConfig => "swiftlint-config",
            Lang::DartAnalysis => "dart-analysis",
            Lang::CMake => "cmake",
            Lang::ClangTidyConfig => "clang-tidy-config",
        }
    }

    /// Whether this is a linter/build configuration format rather than a
    /// programming language. In a configuration format string contents are
    /// the configuration, so the lexer keeps them in the code view.
    pub fn is_config(self) -> bool {
        matches!(
            self,
            Lang::Yaml
                | Lang::CiConfig
                | Lang::Dockerfile
                | Lang::Makefile
                | Lang::CargoManifest
                | Lang::ClippyConfig
                | Lang::PythonConfig
                | Lang::JsLintConfig
                | Lang::TsConfig
                | Lang::NodePackage
                | Lang::GolangciConfig
                | Lang::MsBuild
                | Lang::EditorConfig
                | Lang::Gradle
                | Lang::Maven
                | Lang::RubocopConfig
                | Lang::PhpLintConfig
                | Lang::SwiftlintConfig
                | Lang::DartAnalysis
                | Lang::CMake
                | Lang::ClangTidyConfig
        )
    }

    /// Parse a bare file extension (`"rs"`, `".tsx"`, `"Dockerfile"`).
    /// Extensions that name a generic format (`toml`, `json`, `xml`, `ini`)
    /// return `None`: which tool's configuration it is depends on the file
    /// name, so use [`Lang::from_path`].
    pub fn from_extension(ext: &str) -> Option<Self> {
        let ext = ext.trim_start_matches('.').to_ascii_lowercase();
        Some(match ext.as_str() {
            "rs" => Lang::Rust,
            "py" | "pyi" | "pyw" => Lang::Python,
            "c" | "h" => Lang::C,
            "cpp" | "cc" | "cxx" | "c++" | "hpp" | "hh" | "hxx" | "ipp" | "inl" | "tpp" | "cu" => {
                Lang::Cpp
            }
            "cs" | "csx" => Lang::CSharp,
            "java" => Lang::Java,
            "kt" | "kts" => Lang::Kotlin,
            "scala" | "sc" => Lang::Scala,
            "go" => Lang::Go,
            "js" | "jsx" | "mjs" | "cjs" => Lang::JavaScript,
            "ts" | "tsx" | "mts" | "cts" => Lang::TypeScript,
            "rb" | "rake" | "gemspec" | "ru" => Lang::Ruby,
            "php" | "phtml" => Lang::Php,
            "swift" => Lang::Swift,
            "m" | "mm" => Lang::ObjectiveC,
            "dart" => Lang::Dart,
            "lua" => Lang::Lua,
            "pl" | "pm" | "t" => Lang::Perl,
            "r" => Lang::R,
            "jl" => Lang::Julia,
            "hs" | "lhs" => Lang::Haskell,
            "ml" | "mli" => Lang::OCaml,
            "ex" | "exs" => Lang::Elixir,
            "erl" | "hrl" => Lang::Erlang,
            "clj" | "cljs" | "cljc" | "edn" => Lang::Clojure,
            "zig" => Lang::Zig,
            "nim" | "nims" => Lang::Nim,
            "sol" => Lang::Solidity,
            "sh" | "bash" | "zsh" | "ksh" | "dash" | "bats" => Lang::Shell,
            "ps1" | "psm1" | "psd1" => Lang::PowerShell,
            "sql" => Lang::Sql,
            "tf" | "tfvars" | "hcl" => Lang::Terraform,
            "dockerfile" | "containerfile" => Lang::Dockerfile,
            "mk" | "mak" => Lang::Makefile,
            "yml" | "yaml" => Lang::Yaml,
            "csproj" | "vbproj" | "fsproj" | "props" | "targets" | "ruleset" => Lang::MsBuild,
            "gradle" => Lang::Gradle,
            "cmake" => Lang::CMake,
            "neon" => Lang::PhpLintConfig,
            _ => return None,
        })
    }

    /// Parse a language name or common alias (`"c++"`, `"golang"`, `"ts"`,
    /// `"bash"`, `"github-actions"`). Every [`Lang::as_str`] is accepted.
    pub fn from_name(name: &str) -> Option<Self> {
        let lower = name.trim().to_ascii_lowercase();
        if let Some(lang) = Lang::ALL.iter().copied().find(|l| l.as_str() == lower) {
            return Some(lang);
        }
        Some(match lower.as_str() {
            "rs" => Lang::Rust,
            "py" | "python3" | "python2" | "py3" => Lang::Python,
            "c++" | "cxx" | "cc" => Lang::Cpp,
            "c#" | "cs" | "dotnet" => Lang::CSharp,
            "kt" => Lang::Kotlin,
            "golang" => Lang::Go,
            "js" | "jsx" | "node" | "nodejs" | "ecmascript" => Lang::JavaScript,
            "ts" | "tsx" => Lang::TypeScript,
            "rb" => Lang::Ruby,
            "objc" | "objectivec" | "obj-c" | "objective c" => Lang::ObjectiveC,
            "pl" => Lang::Perl,
            "rlang" => Lang::R,
            "jl" => Lang::Julia,
            "hs" => Lang::Haskell,
            "ml" => Lang::OCaml,
            "ex" | "exs" => Lang::Elixir,
            "erl" => Lang::Erlang,
            "clj" | "cljs" | "clojurescript" => Lang::Clojure,
            "sol" => Lang::Solidity,
            "sh" | "bash" | "zsh" | "ksh" | "dash" | "posix" | "shellscript" | "bats" => {
                Lang::Shell
            }
            "pwsh" | "ps1" | "ps" | "posh" => Lang::PowerShell,
            "postgresql" | "postgres" | "mysql" | "sqlite" | "tsql" | "t-sql" | "plsql"
            | "plpgsql" => Lang::Sql,
            "hcl" | "tf" => Lang::Terraform,
            "docker" | "containerfile" => Lang::Dockerfile,
            "make" | "gnumake" | "mk" => Lang::Makefile,
            "yml" => Lang::Yaml,
            "ci" | "github-actions" | "gitlab-ci" | "azure-pipelines" | "circleci"
            | "jenkinsfile" | "pre-commit" => Lang::CiConfig,
            "cargo" | "cargo.toml" => Lang::CargoManifest,
            "clippy.toml" | "clippy" => Lang::ClippyConfig,
            "pyproject" | "pyproject.toml" | "setup.cfg" | "tox.ini" | "mypy.ini" | "flake8" => {
                Lang::PythonConfig
            }
            "eslintrc" | "eslint" | "biome" | ".eslintrc" => Lang::JsLintConfig,
            "tsconfig.json" | "jsconfig" => Lang::TsConfig,
            "package.json" | "npm" | "jest-config" => Lang::NodePackage,
            "golangci" | "golangci-lint" => Lang::GolangciConfig,
            "csproj" | "msbuild-project" => Lang::MsBuild,
            ".editorconfig" => Lang::EditorConfig,
            "groovy-gradle" | "gradle-kts" => Lang::Gradle,
            "pom" | "pom.xml" | "checkstyle" | "pmd" => Lang::Maven,
            "rubocop" | ".rubocop.yml" => Lang::RubocopConfig,
            "phpstan" | "psalm" | "phpcs" => Lang::PhpLintConfig,
            "swiftlint" => Lang::SwiftlintConfig,
            "analysis_options" | "analysis_options.yaml" => Lang::DartAnalysis,
            "cmakelists" | "cmakelists.txt" => Lang::CMake,
            "clang-tidy" | ".clang-tidy" => Lang::ClangTidyConfig,
            _ => return None,
        })
    }

    /// Read a `#!` line: `#!/bin/bash`, `#!/usr/bin/env -S python3 -u`,
    /// `#!/usr/bin/env ruby`. Version suffixes are ignored (`python3.12`).
    pub fn from_shebang(first_line: &str) -> Option<Self> {
        let rest = first_line
            .trim_start_matches('\u{feff}')
            .strip_prefix("#!")?;
        let mut words = rest.split_whitespace();
        let base = |p: &'_ str| p.rsplit('/').next().unwrap_or(p).to_string();
        let mut program = base(words.next()?);
        if program == "env" {
            program = base(words.find(|w| !w.starts_with('-') && !w.contains('='))?);
        }
        let stem = program.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
        Some(match stem {
            "sh" | "bash" | "zsh" | "ksh" | "dash" | "ash" | "mksh" | "yash" | "busybox"
            | "bats" => Lang::Shell,
            "python" | "pypy" => Lang::Python,
            "node" | "nodejs" | "bun" | "qjs" => Lang::JavaScript,
            "deno" | "ts-node" | "tsx" => Lang::TypeScript,
            "ruby" | "jruby" | "truffleruby" => Lang::Ruby,
            "perl" => Lang::Perl,
            "php" => Lang::Php,
            "lua" | "luajit" => Lang::Lua,
            "Rscript" => Lang::R,
            "julia" => Lang::Julia,
            "pwsh" | "powershell" => Lang::PowerShell,
            "elixir" => Lang::Elixir,
            "escript" => Lang::Erlang,
            "runghc" | "runhaskell" | "stack" | "cabal" => Lang::Haskell,
            "ocaml" => Lang::OCaml,
            "swift" => Lang::Swift,
            "dart" => Lang::Dart,
            "scala" | "amm" | "scala-cli" => Lang::Scala,
            "kotlin" | "kscript" => Lang::Kotlin,
            "bb" | "clojure" | "clj" => Lang::Clojure,
            "make" | "gmake" => Lang::Makefile,
            "rust-script" | "cargo" => Lang::Rust,
            "nim" => Lang::Nim,
            _ => return None,
        })
    }

    /// Classify a path by file name first (configuration formats, CI
    /// pipelines, `Dockerfile`, `Makefile`), then by extension.
    pub fn from_path(path: &str) -> Option<Self> {
        let norm = path.replace('\\', "/");
        let lower = norm.to_ascii_lowercase();
        let base = lower.rsplit('/').next().unwrap_or(&lower);
        let is_yaml = base.ends_with(".yml") || base.ends_with(".yaml");
        let under = |dir: &str| lower.starts_with(dir) || lower.contains(&format!("/{dir}"));

        if base == "dockerfile"
            || base == "containerfile"
            || base.starts_with("dockerfile.")
            || base.ends_with(".dockerfile")
            || base.ends_with(".containerfile")
            || base == ".hadolint.yaml"
            || base == ".hadolint.yml"
        {
            return Some(Lang::Dockerfile);
        }
        if matches!(base, "makefile" | "gnumakefile" | "justfile" | ".justfile") {
            return Some(Lang::Makefile);
        }
        if (is_yaml
            && (under(".github/workflows/")
                || under(".circleci/")
                || under(".buildkite/")
                || under(".woodpecker/")
                || under(".gitlab/ci/")))
            || matches!(
                base,
                ".gitlab-ci.yml"
                    | ".travis.yml"
                    | "azure-pipelines.yml"
                    | "azure-pipelines.yaml"
                    | "bitbucket-pipelines.yml"
                    | ".pre-commit-config.yaml"
                    | ".pre-commit-config.yml"
                    | "appveyor.yml"
                    | ".appveyor.yml"
                    | ".drone.yml"
                    | ".woodpecker.yml"
                    | "cloudbuild.yaml"
                    | "cloudbuild.yml"
                    | "codemagic.yaml"
                    | "jenkinsfile"
            )
        {
            return Some(Lang::CiConfig);
        }
        if base == "cargo.toml"
            || lower.ends_with(".cargo/config.toml")
            || lower.ends_with(".cargo/config")
        {
            return Some(Lang::CargoManifest);
        }
        if base == "clippy.toml" || base == ".clippy.toml" {
            return Some(Lang::ClippyConfig);
        }
        if matches!(
            base,
            "pyproject.toml"
                | "setup.cfg"
                | "tox.ini"
                | ".flake8"
                | "mypy.ini"
                | ".mypy.ini"
                | ".pylintrc"
                | "pylintrc"
                | "ruff.toml"
                | ".ruff.toml"
                | "pytest.ini"
                | ".coveragerc"
                | "pyrightconfig.json"
                | ".bandit"
        ) {
            return Some(Lang::PythonConfig);
        }
        if base.starts_with(".eslintrc")
            || base.starts_with("eslint.config.")
            || matches!(
                base,
                "biome.json" | "biome.jsonc" | ".jshintrc" | ".solhint.json" | ".oxlintrc.json"
            )
        {
            return Some(Lang::JsLintConfig);
        }
        if base == "jsconfig.json" || (base.starts_with("tsconfig") && base.ends_with(".json")) {
            return Some(Lang::TsConfig);
        }
        if base == "package.json"
            || base.starts_with("jest.config.")
            || base.starts_with("vitest.config.")
            || base.starts_with(".mocharc")
            || base == "karma.conf.js"
        {
            return Some(Lang::NodePackage);
        }
        if base.starts_with(".golangci.") {
            return Some(Lang::GolangciConfig);
        }
        if base == ".editorconfig" || base == ".globalconfig" {
            return Some(Lang::EditorConfig);
        }
        if matches!(
            base,
            "build.gradle" | "build.gradle.kts" | "settings.gradle" | "settings.gradle.kts"
        ) {
            return Some(Lang::Gradle);
        }
        if base == "pom.xml"
            || (base.ends_with(".xml")
                && (base.contains("checkstyle")
                    || base.contains("suppressions")
                    || base.starts_with("pmd")
                    || base.contains("spotbugs")
                    || base.contains("findbugs")))
        {
            return Some(Lang::Maven);
        }
        if matches!(base, ".rubocop.yml" | ".rubocop_todo.yml" | ".standard.yml") {
            return Some(Lang::RubocopConfig);
        }
        if base.starts_with("phpstan.")
            || base.starts_with("psalm.xml")
            || base.starts_with("phpcs.xml")
            || base.starts_with(".phpcs.xml")
            || base.starts_with("phpunit.xml")
        {
            return Some(Lang::PhpLintConfig);
        }
        if base == ".swiftlint.yml" || base == ".swiftlint.yaml" {
            return Some(Lang::SwiftlintConfig);
        }
        if base == "analysis_options.yaml" {
            return Some(Lang::DartAnalysis);
        }
        if base == "cmakelists.txt" {
            return Some(Lang::CMake);
        }
        if base == ".clang-tidy" {
            return Some(Lang::ClangTidyConfig);
        }
        if matches!(
            base,
            ".shellcheckrc" | "shellcheckrc" | ".bashrc" | ".zshrc" | ".profile"
        ) {
            return Some(Lang::Shell);
        }
        if base == ".sqlfluff" {
            return Some(Lang::Sql);
        }
        if base == ".luacheckrc" {
            return Some(Lang::Lua);
        }
        if base == ".perlcriticrc" {
            return Some(Lang::Perl);
        }
        if base.starts_with(".yamllint") {
            return Some(Lang::Yaml);
        }
        match base.rsplit_once('.') {
            Some((stem, ext)) if !stem.is_empty() => Lang::from_extension(ext),
            _ => None,
        }
    }

    /// The [`SnippetLanguage`] family cheat.rs scans this language as.
    ///
    /// The six shipped families keep their meaning, and the mapping agrees
    /// with [`crate::langdata::snippet_language`] (Go scans as C, Java and C#
    /// as C++, PHP as JS/TS). Everything else is [`SnippetLanguage::Other`]:
    /// a Kotlin trace declared as C++ would lex correctly but score as a
    /// different language than declared and trip the mixing check, so the
    /// language-specific coverage for these comes from [`scan`] instead.
    pub fn snippet_family(self) -> SnippetLanguage {
        match self {
            Lang::Rust => SnippetLanguage::Rust,
            Lang::Python => SnippetLanguage::Python,
            Lang::C | Lang::ObjectiveC | Lang::Go => SnippetLanguage::C,
            Lang::Cpp | Lang::Java | Lang::CSharp => SnippetLanguage::Cpp,
            Lang::JavaScript | Lang::TypeScript | Lang::Php => SnippetLanguage::JsTs,
            _ => SnippetLanguage::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// Rule vocabulary
// ---------------------------------------------------------------------------

/// What a hit does to the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolyKind {
    /// A linter, compiler warning or security scanner silenced in source.
    LintSuppression,
    /// A test switched off: skipped, hidden, compiled out, commented out.
    TestSkip,
    /// A test that still runs but can no longer fail the gate: expected
    /// failures, focus that hides siblings, `|| true`, coverage exclusion.
    TestWeakening,
    /// A hole presented as an implementation.
    Stub,
    /// A linter, type checker or gate weakened in its configuration.
    ConfigWeakening,
    /// The type checker told to trust the code instead of checking it.
    TypeCheckBypass,
    /// A runtime safety net removed: errors discarded, strict mode off,
    /// overflow checks off.
    UnsafeEscape,
}

impl PolyKind {
    pub const ALL: &'static [PolyKind] = &[
        PolyKind::LintSuppression,
        PolyKind::TestSkip,
        PolyKind::TestWeakening,
        PolyKind::Stub,
        PolyKind::ConfigWeakening,
        PolyKind::TypeCheckBypass,
        PolyKind::UnsafeEscape,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            PolyKind::LintSuppression => "lint-suppression",
            PolyKind::TestSkip => "test-skip",
            PolyKind::TestWeakening => "test-weakening",
            PolyKind::Stub => "stub",
            PolyKind::ConfigWeakening => "config-weakening",
            PolyKind::TypeCheckBypass => "type-check-bypass",
            PolyKind::UnsafeEscape => "unsafe-escape",
        }
    }
}

/// Which view of a line a rule's pattern is matched against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// The start of a comment (after `*`, `/`, `!`, `-` decoration), or of a
    /// later segment opened by the same line-comment marker.
    Directive,
    /// Anywhere inside a comment: for tools that search the whole comment.
    Comment,
    /// Code with comments removed; in programming languages string contents
    /// are blanked too (a directive spelled inside a string is data).
    Code,
    /// Code with comments removed and string contents kept: for patterns
    /// whose payload is a string (a shell argument, a JSON value).
    Text,
}

/// The outcome of a structural [`Pat::Check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckHit {
    No,
    Yes,
    /// The pattern is present and carries its own justification (a Rust
    /// attribute's `reason = "..."`).
    Justified,
}

/// A structural detector too shaped for a needle; its name is its contract.
pub type CheckFn = fn(&Doc<'_>, usize) -> CheckHit;

/// How a rule recognizes its text. In every needle a space matches any run
/// of whitespace, including none (`"# type: ignore"` also matches
/// `#type:ignore`).
#[derive(Debug, Clone, Copy)]
pub enum Pat {
    /// The needle.
    Has(&'static str),
    /// Any of the needles.
    HasAny(&'static [&'static str]),
    /// The needle, with identifier boundaries at edges that are identifier
    /// characters (`t.Skip(` but not `it.Skip(`).
    Word(&'static str),
    /// Any of the needles, as [`Pat::Word`].
    WordAny(&'static [&'static str]),
    /// The site's first non-blank text is one of the needles, as
    /// [`Pat::Word`] on its right edge.
    Lead(&'static [&'static str]),
    /// Each needle in order on the same line.
    Seq(&'static [&'static str]),
    /// Each needle in order, as [`Pat::Word`].
    SeqWord(&'static [&'static str]),
    /// A command-line flag: bounded by whitespace, quotes, `=`, `,`,
    /// brackets or the line edge. Matched exactly.
    Flag(&'static str),
    /// A callee in the code view (as [`Pat::Word`]) with a string literal on
    /// the same line containing one of the phrases (case-insensitive).
    Lit(&'static str, &'static [&'static str]),
    /// A structural detector.
    Check(CheckFn),
}

/// Whether, and how, a written justification downgrades a hit to a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waiver {
    /// Never downgraded.
    Never,
    /// cheat.rs policy: `reason` or `audit-allow:` followed by words, in a
    /// comment on the line or on a comment-only line directly above.
    Reason,
    /// [`Waiver::Reason`], or the tool's own justification slot: two or more
    /// words of three or more letters after this separator on the line, after
    /// the match (empty separator: anywhere after the match).
    Native(&'static str),
}

/// Where escalation markers are looked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// The hit's line and the next non-blank code line.
    Near,
    /// Anywhere in the text.
    File,
}

/// A severity raise when context shows the pattern touches tests: a coverage
/// exclusion on a helper is a note, the same exclusion on a test is evasion.
#[derive(Debug, Clone, Copy)]
pub struct Escalate {
    pub scope: Scope,
    pub markers: &'static [&'static str],
    pub to: Severity,
}

/// One catalog entry. Plain data: [`rules`] returns these, and every field a
/// report shows comes from here.
#[derive(Debug, Clone, Copy)]
pub struct Rule {
    /// Stable identifier, `<lang>.<name>`.
    pub id: &'static str,
    pub kind: PolyKind,
    pub severity: Severity,
    /// The tool whose directive or configuration this is.
    pub tool: Option<&'static str>,
    pub site: Site,
    pub pat: Pat,
    /// ASCII case-insensitive matching (pattern, `requires`, `except`).
    pub ci: bool,
    pub waiver: Waiver,
    /// When non-empty, the line's text view must contain one of these as a
    /// word (`|| true` counts only after a test or lint command).
    pub requires: &'static [&'static str],
    /// The rule does not fire on a line whose raw text contains one of these
    /// (`skipif` on `sys.platform` is platform gating).
    pub except: &'static [&'static str],
    pub escalate: Option<Escalate>,
    /// cheat.rs function that also detects this on the gate side.
    pub overlap: Option<&'static str>,
    /// Why this counts as cheating rather than as ordinary code.
    pub because: &'static str,
}

impl Rule {
    const fn tool(mut self, tool: &'static str) -> Self {
        self.tool = Some(tool);
        self
    }
    const fn ci(mut self) -> Self {
        self.ci = true;
        self
    }
    const fn waiver(mut self, waiver: Waiver) -> Self {
        self.waiver = waiver;
        self
    }
    const fn native(self, sep: &'static str) -> Self {
        self.waiver(Waiver::Native(sep))
    }
    const fn requires(mut self, words: &'static [&'static str]) -> Self {
        self.requires = words;
        self
    }
    const fn except(mut self, needles: &'static [&'static str]) -> Self {
        self.except = needles;
        self
    }
    const fn near(mut self, markers: &'static [&'static str], to: Severity) -> Self {
        self.escalate = Some(Escalate {
            scope: Scope::Near,
            markers,
            to,
        });
        self
    }
    const fn in_file(mut self, markers: &'static [&'static str], to: Severity) -> Self {
        self.escalate = Some(Escalate {
            scope: Scope::File,
            markers,
            to,
        });
        self
    }
    const fn overlap(mut self, function: &'static str) -> Self {
        self.overlap = Some(function);
        self
    }
}

/// One finding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hit {
    pub kind: PolyKind,
    pub lang: Lang,
    pub severity: Severity,
    /// 1-based line number.
    pub line: Option<usize>,
    /// The trimmed source line, truncated for display.
    pub evidence: String,
    pub because: &'static str,
    pub tool: Option<&'static str>,
    /// The [`Rule::id`] that produced this hit.
    pub rule: &'static str,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// A lexed text: one [`Line`] per source line.
#[derive(Debug, Clone)]
pub struct Doc<'a> {
    pub lang: Lang,
    pub lines: Vec<Line<'a>>,
}

/// One source line in three byte-aligned views: every offset into `code` or
/// `text` is the same offset into `raw`.
#[derive(Debug, Clone)]
pub struct Line<'a> {
    /// 1-based line number.
    pub no: usize,
    pub raw: &'a str,
    /// Comments blanked; string contents blanked too unless the language is a
    /// configuration format (quotes are kept).
    pub code: String,
    /// Comments blanked, string contents kept.
    pub text: String,
    /// The comment segments on this line.
    pub comments: Vec<Seg>,
    /// Byte ranges of string-literal contents on this line.
    pub lits: Vec<(usize, usize)>,
}

/// A comment segment: `raw[start..]` opens it, `raw[body..end]` is its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seg {
    pub start: usize,
    pub body: usize,
    pub end: usize,
    /// The line-comment marker that opened it; `None` for block comments.
    pub marker: Option<&'static str>,
}

impl Line<'_> {
    /// The text of every comment segment on the line, space-joined.
    pub fn comment_text(&self) -> String {
        self.comments
            .iter()
            .map(|s| self.raw.get(s.body..s.end).unwrap_or(""))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A line with comments and no code.
    pub fn is_comment_only(&self) -> bool {
        self.code.trim().is_empty() && !self.comments.is_empty()
    }
}

/// Where a line-comment marker may open a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pos {
    Anywhere,
    /// After whitespace, `;|&()` or the line start (shell `$#`, YAML `a#b`).
    WordStart,
    /// Only as the line's first non-blank text (Dockerfile, INI `;`).
    LineStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sq {
    No,
    /// `'...'` with escapes.
    Str,
    /// `'...'` without escapes (shell, SQL, YAML, TOML literal strings).
    Raw,
    /// `'x'` char literals; any other `'` is code (lifetimes, primes).
    Char,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bt {
    No,
    /// JS template literal: escapes, may span lines.
    Str,
    /// Go raw string: no escapes, may span lines.
    Raw,
    /// Quoted identifier or command: no escapes, one line.
    Line,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Here {
    No,
    Shell,
    Ruby,
    Perl,
    Php,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pod {
    No,
    /// `=pod` / `=head1` ... `=cut`.
    Perl,
    /// `=begin` ... `=end`.
    Ruby,
}

/// A language's lexical shape: just enough to tell comments and strings from
/// code.
#[derive(Debug, Clone, Copy)]
struct Syntax {
    /// Line-comment markers: marker, where it may appear, and a following
    /// byte that cancels it (PHP `#[` is an attribute; 0 = none).
    line: &'static [(&'static str, Pos, u8)],
    block: &'static [(&'static str, &'static str)],
    nest: bool,
    dq: bool,
    sq: Sq,
    bt: Bt,
    triple: &'static [u8],
    /// Ordinary quoted strings may span lines.
    ml: bool,
    here: Here,
    escape: u8,
    keep_strings: bool,
    /// A quote opens a string only at the start of a token (YAML `don't`).
    quote_word_start: bool,
    rust_raw: bool,
    cpp_raw: bool,
    cs_verbatim: bool,
    lua_long: bool,
    zig_lines: bool,
    hs_dashes: bool,
    /// A byte that introduces a one-character literal (`$"` in Erlang,
    /// `?"` in Elixir, `\"` in Clojure); 0 = none.
    char_prefix: u8,
    pod: Pod,
    ps_here: bool,
}

const BASE: Syntax = Syntax {
    line: &[],
    block: &[],
    nest: false,
    dq: true,
    sq: Sq::No,
    bt: Bt::No,
    triple: b"",
    ml: false,
    here: Here::No,
    escape: b'\\',
    keep_strings: false,
    quote_word_start: false,
    rust_raw: false,
    cpp_raw: false,
    cs_verbatim: false,
    lua_long: false,
    zig_lines: false,
    hs_dashes: false,
    char_prefix: 0,
    pod: Pod::No,
    ps_here: false,
};

const SLASH: &[(&str, Pos, u8)] = &[("//", Pos::Anywhere, 0)];
const HASH: &[(&str, Pos, u8)] = &[("#", Pos::Anywhere, 0)];
const HASH_WORD: &[(&str, Pos, u8)] = &[("#", Pos::WordStart, 0)];
const C_BLOCK: &[(&str, &str)] = &[("/*", "*/")];

const fn syntax(lang: Lang) -> Syntax {
    match lang {
        Lang::Rust => Syntax {
            line: SLASH,
            block: C_BLOCK,
            nest: true,
            sq: Sq::Char,
            ml: true,
            rust_raw: true,
            ..BASE
        },
        Lang::C | Lang::Cpp | Lang::ObjectiveC => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Char,
            cpp_raw: true,
            ..BASE
        },
        Lang::CSharp => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Char,
            triple: b"\"",
            cs_verbatim: true,
            ..BASE
        },
        Lang::Java => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Char,
            triple: b"\"",
            ..BASE
        },
        Lang::Kotlin | Lang::Scala => Syntax {
            line: SLASH,
            block: C_BLOCK,
            nest: true,
            sq: Sq::Char,
            bt: Bt::Line,
            triple: b"\"",
            ..BASE
        },
        Lang::Go => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Char,
            bt: Bt::Raw,
            ..BASE
        },
        Lang::JavaScript | Lang::TypeScript => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Str,
            bt: Bt::Str,
            ..BASE
        },
        Lang::Python => Syntax {
            line: HASH,
            sq: Sq::Str,
            triple: b"\"'",
            ..BASE
        },
        Lang::Ruby => Syntax {
            line: HASH,
            sq: Sq::Str,
            bt: Bt::Line,
            ml: true,
            here: Here::Ruby,
            pod: Pod::Ruby,
            ..BASE
        },
        Lang::Php => Syntax {
            line: &[("//", Pos::Anywhere, 0), ("#", Pos::Anywhere, b'[')],
            block: C_BLOCK,
            sq: Sq::Str,
            bt: Bt::Line,
            ml: true,
            here: Here::Php,
            ..BASE
        },
        Lang::Swift => Syntax {
            line: SLASH,
            block: C_BLOCK,
            nest: true,
            bt: Bt::Line,
            triple: b"\"",
            ..BASE
        },
        Lang::Dart => Syntax {
            line: SLASH,
            block: C_BLOCK,
            nest: true,
            sq: Sq::Str,
            triple: b"\"'",
            ..BASE
        },
        Lang::Lua => Syntax {
            line: &[("--", Pos::Anywhere, 0)],
            sq: Sq::Str,
            lua_long: true,
            ..BASE
        },
        Lang::Perl => Syntax {
            line: HASH_WORD,
            sq: Sq::Str,
            bt: Bt::Line,
            ml: true,
            here: Here::Perl,
            pod: Pod::Perl,
            ..BASE
        },
        Lang::R => Syntax {
            line: HASH,
            sq: Sq::Str,
            bt: Bt::Line,
            ml: true,
            ..BASE
        },
        Lang::Julia => Syntax {
            line: HASH,
            block: &[("#=", "=#")],
            nest: true,
            sq: Sq::Char,
            bt: Bt::Line,
            triple: b"\"",
            ..BASE
        },
        Lang::Haskell => Syntax {
            line: &[("--", Pos::Anywhere, 0)],
            block: &[("{-", "-}")],
            nest: true,
            sq: Sq::Char,
            hs_dashes: true,
            ..BASE
        },
        Lang::OCaml => Syntax {
            block: &[("(*", "*)")],
            nest: true,
            sq: Sq::Char,
            ml: true,
            ..BASE
        },
        Lang::Elixir => Syntax {
            line: HASH,
            sq: Sq::Str,
            triple: b"\"'",
            ml: true,
            char_prefix: b'?',
            ..BASE
        },
        Lang::Erlang => Syntax {
            line: &[("%", Pos::Anywhere, 0)],
            sq: Sq::Str,
            ml: true,
            char_prefix: b'$',
            ..BASE
        },
        Lang::Clojure => Syntax {
            line: &[(";", Pos::Anywhere, 0)],
            ml: true,
            char_prefix: b'\\',
            ..BASE
        },
        Lang::Zig => Syntax {
            line: SLASH,
            sq: Sq::Char,
            zig_lines: true,
            ..BASE
        },
        Lang::Nim => Syntax {
            line: HASH,
            block: &[("#[", "]#")],
            nest: true,
            sq: Sq::Char,
            bt: Bt::Line,
            triple: b"\"",
            ..BASE
        },
        Lang::Solidity => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Str,
            ..BASE
        },
        Lang::Shell => Syntax {
            line: HASH_WORD,
            sq: Sq::Raw,
            ml: true,
            here: Here::Shell,
            ..BASE
        },
        Lang::PowerShell => Syntax {
            line: HASH_WORD,
            block: &[("<#", "#>")],
            sq: Sq::Raw,
            ml: true,
            escape: b'`',
            ps_here: true,
            ..BASE
        },
        Lang::Sql => Syntax {
            line: &[("--", Pos::Anywhere, 0)],
            block: C_BLOCK,
            sq: Sq::Raw,
            bt: Bt::Line,
            ml: true,
            ..BASE
        },
        Lang::Terraform => Syntax {
            line: &[("#", Pos::Anywhere, 0), ("//", Pos::Anywhere, 0)],
            block: C_BLOCK,
            here: Here::Shell,
            ..BASE
        },
        Lang::Dockerfile => Syntax {
            line: &[("#", Pos::LineStart, 0)],
            sq: Sq::Raw,
            here: Here::Shell,
            keep_strings: true,
            ..BASE
        },
        Lang::Makefile => Syntax {
            line: HASH,
            sq: Sq::Raw,
            keep_strings: true,
            quote_word_start: true,
            ..BASE
        },
        Lang::Yaml
        | Lang::GolangciConfig
        | Lang::RubocopConfig
        | Lang::SwiftlintConfig
        | Lang::DartAnalysis
        | Lang::ClangTidyConfig => Syntax {
            line: HASH_WORD,
            sq: Sq::Raw,
            keep_strings: true,
            quote_word_start: true,
            ..BASE
        },
        Lang::CiConfig => Syntax {
            line: &[("#", Pos::WordStart, 0), ("//", Pos::WordStart, 0)],
            sq: Sq::Raw,
            keep_strings: true,
            quote_word_start: true,
            ..BASE
        },
        Lang::CargoManifest | Lang::ClippyConfig | Lang::PythonConfig | Lang::EditorConfig => {
            Syntax {
                line: &[
                    ("#", Pos::Anywhere, 0),
                    (";", Pos::LineStart, 0),
                    ("//", Pos::WordStart, 0),
                ],
                sq: Sq::Raw,
                triple: b"\"'",
                keep_strings: true,
                quote_word_start: true,
                ..BASE
            }
        }
        Lang::TsConfig => Syntax {
            line: SLASH,
            block: C_BLOCK,
            keep_strings: true,
            ..BASE
        },
        Lang::JsLintConfig | Lang::NodePackage => Syntax {
            line: &[("//", Pos::Anywhere, 0), ("#", Pos::WordStart, 0)],
            block: C_BLOCK,
            sq: Sq::Str,
            bt: Bt::Str,
            keep_strings: true,
            ..BASE
        },
        Lang::MsBuild | Lang::Maven => Syntax {
            block: &[("<!--", "-->")],
            keep_strings: true,
            ..BASE
        },
        Lang::PhpLintConfig => Syntax {
            line: HASH_WORD,
            block: &[("<!--", "-->")],
            sq: Sq::Raw,
            keep_strings: true,
            quote_word_start: true,
            ..BASE
        },
        Lang::Gradle => Syntax {
            line: SLASH,
            block: C_BLOCK,
            sq: Sq::Str,
            triple: b"\"'",
            keep_strings: true,
            ..BASE
        },
        Lang::CMake => Syntax {
            line: HASH,
            block: &[("#[[", "]]")],
            ml: true,
            keep_strings: true,
            ..BASE
        },
    }
}

/// Lexer state carried across lines.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Code,
    Block {
        open: &'static str,
        close: &'static str,
        depth: usize,
    },
    LuaLong {
        level: usize,
        comment: bool,
    },
    Str(StrMode),
    Here {
        delim: String,
        squish: bool,
        php: bool,
    },
    Pod(&'static str),
    PsHere(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StrMode {
    Quote {
        q: u8,
        escapes: bool,
        multiline: bool,
    },
    Triple(u8),
    RustRaw(usize),
    CppRaw(String),
}

/// The byte at `i`, or 0 past the end (0 matches no marker).
fn at(b: &[u8], i: usize) -> u8 {
    b.get(i).copied().unwrap_or(0)
}

fn at_prev(b: &[u8], i: usize) -> u8 {
    i.checked_sub(1).map_or(0, |p| at(b, p))
}

fn starts_with_at(b: &[u8], i: usize, s: &str) -> bool {
    b.get(i..).is_some_and(|t| t.starts_with(s.as_bytes()))
}

/// Identifier byte; non-ASCII counts, since identifiers may be Unicode.
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

/// Length of the UTF-8 character starting at `i` (0 past the end).
fn char_len(b: &[u8], i: usize) -> usize {
    match b.get(i) {
        None => 0,
        Some(&c) if c < 0x80 => 1,
        Some(&c) if c >= 0xF0 => 4,
        Some(&c) if c >= 0xE0 => 3,
        Some(_) => 2,
    }
}

struct LineLex<'a> {
    raw: &'a str,
    code: Vec<u8>,
    text: Vec<u8>,
    comments: Vec<Seg>,
    lits: Vec<(usize, usize)>,
    keep_strings: bool,
}

fn blank(v: &mut [u8], from: usize, to: usize) {
    for byte in v.iter_mut().take(to).skip(from) {
        *byte = b' ';
    }
}

impl<'a> LineLex<'a> {
    fn new(raw: &'a str, keep_strings: bool) -> Self {
        Self {
            raw,
            code: raw.as_bytes().to_vec(),
            text: raw.as_bytes().to_vec(),
            comments: Vec::new(),
            lits: Vec::new(),
            keep_strings,
        }
    }

    /// Record a comment: `[start, until)` is blanked from both views and
    /// `[body, end)` is its text.
    fn comment(
        &mut self,
        start: usize,
        body: usize,
        end: usize,
        until: usize,
        marker: Option<&'static str>,
    ) {
        blank(&mut self.code, start, until);
        blank(&mut self.text, start, until);
        self.comments.push(Seg {
            start,
            body: body.min(end),
            end,
            marker,
        });
    }

    fn literal(&mut self, from: usize, to: usize) {
        if from < to {
            self.lits.push((from, to));
            if !self.keep_strings {
                blank(&mut self.code, from, to);
            }
        }
    }

    fn finish(self, no: usize) -> Line<'a> {
        // Blanking always covers whole characters (ranges start and end at
        // ASCII delimiters or the line edge), so both views stay UTF-8.
        let to_string =
            |v: Vec<u8>| String::from_utf8(v).unwrap_or_else(|e| " ".repeat(e.into_bytes().len()));
        Line {
            no,
            raw: self.raw,
            code: to_string(self.code),
            text: to_string(self.text),
            comments: self.comments,
            lits: self.lits,
        }
    }
}

fn pos_ok(b: &[u8], i: usize, pos: Pos) -> bool {
    match pos {
        Pos::Anywhere => true,
        Pos::WordStart => {
            let p = at_prev(b, i);
            i == 0 || p.is_ascii_whitespace() || b";|&()".contains(&p)
        }
        Pos::LineStart => b
            .get(..i)
            .is_some_and(|t| t.iter().all(u8::is_ascii_whitespace)),
    }
}

fn quote_ok(syn: &Syntax, b: &[u8], i: usize) -> bool {
    if !syn.quote_word_start || i == 0 {
        return true;
    }
    let p = at_prev(b, i);
    p.is_ascii_whitespace() || b":[{,(=".contains(&p)
}

/// Haskell: `--` followed by a symbol character is an operator (`-->`).
fn hs_operator(b: &[u8], i: usize) -> bool {
    let mut j = i + 2;
    while at(b, j) == b'-' {
        j += 1;
    }
    let c = at(b, j);
    c != 0 && b"!#$%&*+./<=>?@\\^|~:".contains(&c)
}

/// `'x'`, `'\n'`, `'\u{1F600}'`: the index just past the closing quote.
fn char_literal(b: &[u8], i: usize) -> Option<usize> {
    if at(b, i + 1) == b'\\' {
        let limit = (i + 14).min(b.len());
        return (i + 3..limit).find(|&j| at(b, j) == b'\'').map(|j| j + 1);
    }
    let len = char_len(b, i + 1);
    (len > 0 && at(b, i + 1) != b'\'' && at(b, i + 1 + len) == b'\'').then_some(i + 2 + len)
}

/// Lua long bracket `[`, `=`*, `[` at `i`: its level.
fn long_bracket(b: &[u8], i: usize) -> Option<usize> {
    if at(b, i) != b'[' {
        return None;
    }
    let mut level = 0;
    while at(b, i + 1 + level) == b'=' {
        level += 1;
    }
    (at(b, i + 1 + level) == b'[').then_some(level)
}

/// `r"`, `r#"`, `br"`, `cr##"`: (opener length, hash count).
fn rust_raw_open(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let mut k = i;
    if matches!(at(b, k), b'b' | b'c') {
        k += 1;
    }
    if at(b, k) != b'r' {
        return None;
    }
    k += 1;
    let mut hashes = 0;
    while at(b, k) == b'#' {
        hashes += 1;
        k += 1;
    }
    (at(b, k) == b'"').then_some((k + 1 - i, hashes))
}

/// C++ `R"delim(`: (closing sequence `)delim"`, opener length).
fn cpp_raw_open(b: &[u8], i: usize) -> Option<(String, usize)> {
    let mut k = i + 2;
    let start = k;
    while k < b.len() && k - start <= 16 && !b"( )\\\t\"".contains(&at(b, k)) {
        k += 1;
    }
    if at(b, k) != b'(' {
        return None;
    }
    let delim = String::from_utf8_lossy(b.get(start..k)?).into_owned();
    Some((format!("){delim}\""), k + 1 - i))
}

/// A heredoc delimiter word at `k`: (word, index after it).
fn heredoc_word(b: &[u8], k: usize, bare_upper: bool) -> Option<(String, usize)> {
    let q = at(b, k);
    let quoted = matches!(q, b'\'' | b'"' | b'`');
    let start = if quoted { k + 1 } else { k };
    let first = at(b, start);
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    if !quoted && bare_upper && !first.is_ascii_uppercase() {
        return None;
    }
    let mut s = start;
    while at(b, s).is_ascii_alphanumeric() || at(b, s) == b'_' {
        s += 1;
    }
    let word = String::from_utf8_lossy(b.get(start..s)?).into_owned();
    if quoted {
        if at(b, s) != q {
            return None;
        }
        s += 1;
    }
    Some((word, s))
}

/// A heredoc opener at `i`: the mode for the following lines, and the index
/// just past the opener.
fn heredoc_open(kind: Here, b: &[u8], i: usize) -> Option<(Mode, usize)> {
    let mode = |delim: String, squish: bool, php: bool| Mode::Here { delim, squish, php };
    match kind {
        Here::No => None,
        Here::Shell => {
            if !starts_with_at(b, i, "<<") || at(b, i + 2) == b'<' {
                return None;
            }
            let mut k = i + 2;
            let squish = at(b, k) == b'-';
            if squish {
                k += 1;
            }
            while matches!(at(b, k), b' ' | b'\t') {
                k += 1;
            }
            let (delim, end) = heredoc_word(b, k, false)?;
            Some((mode(delim, squish, false), end))
        }
        Here::Ruby | Here::Perl => {
            if !starts_with_at(b, i, "<<") || at(b, i + 2) == b'<' {
                return None;
            }
            let mut k = i + 2;
            let flag = at(b, k) == b'~' || (kind == Here::Ruby && at(b, k) == b'-');
            if flag {
                k += 1;
            }
            let (delim, end) = heredoc_word(b, k, !flag)?;
            Some((mode(delim, flag, false), end))
        }
        Here::Php => {
            if !starts_with_at(b, i, "<<<") {
                return None;
            }
            let mut k = i + 3;
            while matches!(at(b, k), b' ' | b'\t') {
                k += 1;
            }
            let (delim, end) = heredoc_word(b, k, false)?;
            Some((mode(delim, true, true), end))
        }
    }
}

fn pod_open(pod: Pod, raw: &str) -> Option<&'static str> {
    let b = raw.as_bytes();
    match pod {
        Pod::Perl if at(b, 0) == b'=' && at(b, 1).is_ascii_alphabetic() => Some("=cut"),
        Pod::Ruby
            if raw.starts_with("=begin")
                && raw.as_bytes().get(6).is_none_or(u8::is_ascii_whitespace) =>
        {
            Some("=end")
        }
        _ => None,
    }
}

/// One step in code: returns the next index and, when a construct opens, the
/// mode it opens.
fn code_step(
    syn: &Syntax,
    b: &[u8],
    i: usize,
    lx: &mut LineLex<'_>,
    pending: &mut Option<Mode>,
) -> (usize, Option<Mode>) {
    let n = b.len();
    let c = at(b, i);
    if syn.zig_lines && c == b'\\' && at(b, i + 1) == b'\\' {
        lx.literal(i + 2, n);
        return (n, None);
    }
    for &(open, close) in syn.block {
        if starts_with_at(b, i, open) {
            return (
                i + open.len(),
                Some(Mode::Block {
                    open,
                    close,
                    depth: 1,
                }),
            );
        }
    }
    if syn.lua_long {
        if starts_with_at(b, i, "--") {
            if let Some(level) = long_bracket(b, i + 2) {
                return (
                    i + level + 4,
                    Some(Mode::LuaLong {
                        level,
                        comment: true,
                    }),
                );
            }
        } else if let Some(level) = long_bracket(b, i) {
            return (
                i + level + 2,
                Some(Mode::LuaLong {
                    level,
                    comment: false,
                }),
            );
        }
    }
    for &(marker, pos, cancel) in syn.line {
        if starts_with_at(b, i, marker)
            && pos_ok(b, i, pos)
            && (cancel == 0 || at(b, i + marker.len()) != cancel)
        {
            if syn.hs_dashes && marker == "--" && hs_operator(b, i) {
                return (i + 2, None);
            }
            lx.comment(i, i + marker.len(), n, n, Some(marker));
            return (n, None);
        }
    }
    if syn.rust_raw && matches!(c, b'r' | b'b' | b'c') && !is_ident(at_prev(b, i)) {
        if let Some((len, hashes)) = rust_raw_open(b, i) {
            return (i + len, Some(Mode::Str(StrMode::RustRaw(hashes))));
        }
    }
    if syn.cpp_raw
        && c == b'R'
        && at(b, i + 1) == b'"'
        && (!is_ident(at_prev(b, i)) || matches!(at_prev(b, i), b'8' | b'u' | b'U' | b'L'))
    {
        if let Some((close, len)) = cpp_raw_open(b, i) {
            return (i + len, Some(Mode::Str(StrMode::CppRaw(close))));
        }
    }
    let verbatim = |len: usize| {
        let q = StrMode::Quote {
            q: b'"',
            escapes: false,
            multiline: true,
        };
        (i + len, Some(Mode::Str(q)))
    };
    if syn.cs_verbatim && c == b'@' {
        if at(b, i + 1) == b'"' {
            return verbatim(2);
        }
        if at(b, i + 1) == b'$' && at(b, i + 2) == b'"' {
            return verbatim(3);
        }
    }
    if syn.ps_here
        && c == b'@'
        && matches!(at(b, i + 1), b'"' | b'\'')
        && b.get(i + 2..)
            .is_some_and(|r| r.iter().all(u8::is_ascii_whitespace))
    {
        return (n, Some(Mode::PsHere(at(b, i + 1))));
    }
    if syn.triple.contains(&c) && at(b, i + 1) == c && at(b, i + 2) == c && quote_ok(syn, b, i) {
        return (i + 3, Some(Mode::Str(StrMode::Triple(c))));
    }
    let quote = |q: u8, escapes: bool, multiline: bool| {
        (
            i + 1,
            Some(Mode::Str(StrMode::Quote {
                q,
                escapes,
                multiline,
            })),
        )
    };
    if c == b'"' && syn.dq && quote_ok(syn, b, i) {
        return quote(b'"', true, syn.ml);
    }
    if c == b'\'' {
        match syn.sq {
            Sq::Str if quote_ok(syn, b, i) => return quote(b'\'', true, syn.ml),
            Sq::Raw if quote_ok(syn, b, i) => return quote(b'\'', false, syn.ml),
            Sq::Char => {
                if let Some(end) = char_literal(b, i) {
                    lx.literal(i + 1, end - 1);
                    return (end, None);
                }
            }
            _ => {}
        }
    }
    if c == b'`' {
        match syn.bt {
            Bt::Str => return quote(b'`', true, true),
            Bt::Raw => return quote(b'`', false, true),
            Bt::Line => return quote(b'`', false, false),
            Bt::No => {}
        }
    }
    if c == b'<' {
        if let Some((mode, after)) = heredoc_open(syn.here, b, i) {
            *pending = Some(mode);
            return (after, None);
        }
    }
    if syn.char_prefix != 0 && c == syn.char_prefix && i + 1 < n {
        return (i + 1 + char_len(b, i + 1).max(1), None);
    }
    (i + 1, None)
}

/// Scan a string from `i` to its end or the line's end.
fn str_step(
    syn: &Syntax,
    b: &[u8],
    i: usize,
    s: &StrMode,
    lx: &mut LineLex<'_>,
) -> (usize, Option<Mode>) {
    let n = b.len();
    let mut j = i;
    while j < n {
        let c = at(b, j);
        match s {
            StrMode::Quote { q, escapes, .. } => {
                if *escapes && c == syn.escape {
                    j += 2;
                    continue;
                }
                if c == *q {
                    lx.literal(i, j);
                    return (j + 1, Some(Mode::Code));
                }
            }
            StrMode::Triple(q) => {
                if c == syn.escape {
                    j += 2;
                    continue;
                }
                if c == *q && at(b, j + 1) == *q && at(b, j + 2) == *q {
                    lx.literal(i, j);
                    return (j + 3, Some(Mode::Code));
                }
            }
            StrMode::RustRaw(hashes) => {
                if c == b'"' && (1..=*hashes).all(|h| at(b, j + h) == b'#') {
                    lx.literal(i, j);
                    return (j + 1 + hashes, Some(Mode::Code));
                }
            }
            StrMode::CppRaw(close) => {
                if starts_with_at(b, j, close) {
                    lx.literal(i, j);
                    return (j + close.len(), Some(Mode::Code));
                }
            }
        }
        j += 1;
    }
    lx.literal(i, n);
    (n, None)
}

/// Scan a block comment from `i`; records the segment when it closes.
fn block_step(
    syn: &Syntax,
    b: &[u8],
    i: usize,
    (open, close, depth): (&'static str, &'static str, usize),
    (from, body): (usize, usize),
    lx: &mut LineLex<'_>,
) -> (usize, Option<Mode>) {
    let n = b.len();
    let mut d = depth;
    let mut j = i;
    while j < n {
        if starts_with_at(b, j, close) {
            d = d.saturating_sub(1);
            if d == 0 {
                lx.comment(from, body, j, j + close.len(), None);
                return (j + close.len(), Some(Mode::Code));
            }
            j += close.len();
            continue;
        }
        if syn.nest && starts_with_at(b, j, open) {
            d += 1;
            j += open.len();
            continue;
        }
        j += 1;
    }
    let next = (d != depth).then_some(Mode::Block {
        open,
        close,
        depth: d,
    });
    (n, next)
}

/// Scan a Lua long comment or string from `i`.
fn lua_step(
    b: &[u8],
    i: usize,
    level: usize,
    comment: bool,
    (from, body): (usize, usize),
    lx: &mut LineLex<'_>,
) -> (usize, Option<Mode>) {
    let n = b.len();
    let close = format!("]{}]", "=".repeat(level));
    let mut j = i;
    while j < n {
        if starts_with_at(b, j, &close) {
            if comment {
                lx.comment(from, body, j, j + close.len(), None);
            } else {
                lx.literal(i, j);
            }
            return (j + close.len(), Some(Mode::Code));
        }
        j += 1;
    }
    if !comment {
        lx.literal(i, n);
    }
    (n, None)
}

/// Lex `text` as `lang`. Never fails: unknown constructs are code, and an
/// unterminated one-line string ends with its line.
pub fn lex(text: &str, lang: Lang) -> Doc<'_> {
    let syn = syntax(lang);
    let mut mode = Mode::Code;
    let mut lines = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let mut lx = LineLex::new(raw, syn.keep_strings);
        let b = raw.as_bytes();
        let n = b.len();
        let mut i = 0usize;
        let mut pending: Option<Mode> = None;
        let whole_line = match &mode {
            Mode::Pod(close) => {
                let close: &str = close;
                lx.comment(0, 0, n, n, None);
                if raw.starts_with(close) {
                    mode = Mode::Code;
                }
                true
            }
            Mode::Here { delim, squish, php } => {
                let body = if *squish { raw.trim_start() } else { raw };
                let ends = body.strip_prefix(delim.as_str()).is_some_and(|rest| {
                    if *php {
                        !rest.bytes().next().is_some_and(is_ident)
                    } else {
                        rest.trim().is_empty()
                    }
                });
                if ends {
                    i = n - body.len() + delim.len();
                    mode = Mode::Code;
                    false
                } else {
                    lx.literal(0, n);
                    true
                }
            }
            Mode::PsHere(q) => {
                let t = raw.trim_start();
                if t.as_bytes().first() == Some(q) && t.as_bytes().get(1) == Some(&b'@') {
                    i = n - t.len() + 2;
                    mode = Mode::Code;
                    false
                } else {
                    lx.literal(0, n);
                    true
                }
            }
            Mode::Code => match pod_open(syn.pod, raw) {
                Some(close) => {
                    lx.comment(0, 0, n, n, None);
                    if !(syn.pod == Pod::Perl && raw.starts_with("=cut")) {
                        mode = Mode::Pod(close);
                    }
                    true
                }
                None => false,
            },
            _ => false,
        };
        if whole_line {
            lines.push(lx.finish(idx + 1));
            continue;
        }
        // Where the block comment open on this line began, and its text.
        let mut span = (0usize, 0usize);
        while i < n {
            let (j, change) = match &mode {
                Mode::Code => code_step(&syn, b, i, &mut lx, &mut pending),
                Mode::Block { open, close, depth } => {
                    block_step(&syn, b, i, (*open, *close, *depth), span, &mut lx)
                }
                Mode::LuaLong { level, comment } => lua_step(b, i, *level, *comment, span, &mut lx),
                Mode::Str(s) => str_step(&syn, b, i, s, &mut lx),
                Mode::Here { .. } | Mode::Pod(_) | Mode::PsHere(_) => (n, None),
            };
            if let Some(next) = change {
                let opens_comment = matches!(next, Mode::Block { .. } | Mode::LuaLong { .. });
                if opens_comment && mode == Mode::Code {
                    span = (i, j);
                }
                mode = next;
            }
            i = j.max(i + 1);
        }
        match &mode {
            Mode::Block { .. } | Mode::LuaLong { comment: true, .. } => {
                lx.comment(span.0, span.1, n, n, None);
            }
            Mode::Str(StrMode::Quote {
                multiline: false, ..
            }) => mode = Mode::Code,
            _ => {}
        }
        if mode == Mode::Code {
            if let Some(here) = pending {
                mode = here;
            }
        }
        lines.push(lx.finish(idx + 1));
    }
    Doc { lang, lines }
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// `needle` at exactly `start` in `hay[..limit]`, a space in the needle
/// matching any run of whitespace: the end of the match.
fn match_at(hay: &[u8], start: usize, limit: usize, needle: &str, ci: bool) -> Option<usize> {
    let mut h = start;
    for &nb in needle.as_bytes() {
        if nb == b' ' {
            while h < limit && at(hay, h).is_ascii_whitespace() {
                h += 1;
            }
            continue;
        }
        if h >= limit {
            return None;
        }
        let hb = at(hay, h);
        let same = if ci {
            hb.eq_ignore_ascii_case(&nb)
        } else {
            hb == nb
        };
        if !same {
            return None;
        }
        h += 1;
    }
    Some(h)
}

/// Identifier boundaries at the needle's identifier edges.
fn word_ok(hay: &[u8], s: usize, e: usize, needle: &str, limit: usize) -> bool {
    let nb = needle.as_bytes();
    let left = nb.first().is_some_and(|&c| is_ident(c));
    let right = nb.last().is_some_and(|&c| is_ident(c));
    (!left || s == 0 || !is_ident(at_prev(hay, s)))
        && (!right || e >= limit || !is_ident(at(hay, e)))
}

/// First match of `needle` in `hay[from..limit]`: (start, end).
fn find(
    hay: &[u8],
    from: usize,
    limit: usize,
    needle: &str,
    ci: bool,
    word: bool,
) -> Option<(usize, usize)> {
    let limit = limit.min(hay.len());
    (from..limit).find_map(|s| {
        match_at(hay, s, limit, needle, ci)
            .filter(|&e| !word || word_ok(hay, s, e, needle, limit))
            .map(|e| (s, e))
    })
}

fn first_non_ws(hay: &[u8], from: usize, limit: usize) -> usize {
    let mut p = from;
    while p < limit && at(hay, p).is_ascii_whitespace() {
        p += 1;
    }
    p
}

/// Bytes allowed before and after a [`Pat::Flag`].
const FLAG_LEFT: &[u8] = b" \t\"'=,[(`";
const FLAG_RIGHT: &[u8] = b" \t\"',])`;";

fn flag_find(hay: &[u8], from: usize, limit: usize, flag: &str) -> Option<usize> {
    let limit = limit.min(hay.len());
    let mut s = from;
    while let Some((start, end)) = find(hay, s, limit, flag, false, false) {
        let left = start == 0 || FLAG_LEFT.contains(&at_prev(hay, start));
        let right = end >= limit || FLAG_RIGHT.contains(&at(hay, end));
        if left && right && !flag.contains(' ') {
            return Some(end);
        }
        s = start + 1;
    }
    None
}

/// Match a needle pattern in `hay[from..limit]`; `anchored` pins the first
/// needle to `from` (directive position).
fn pat_in(
    pat: &Pat,
    hay: &[u8],
    from: usize,
    limit: usize,
    ci: bool,
    anchored: bool,
) -> Option<usize> {
    let one = |needle: &str, word: bool, start: usize, anchor: bool| -> Option<usize> {
        if anchor {
            match_at(hay, start, limit, needle, ci)
                .filter(|&e| !word || word_ok(hay, start, e, needle, limit))
        } else {
            find(hay, start, limit, needle, ci, word).map(|(_, e)| e)
        }
    };
    match pat {
        Pat::Has(n) => one(n, false, from, anchored),
        Pat::HasAny(ns) => ns.iter().find_map(|n| one(n, false, from, anchored)),
        Pat::Word(n) => one(n, true, from, anchored),
        Pat::WordAny(ns) => ns.iter().find_map(|n| one(n, true, from, anchored)),
        Pat::Lead(ns) => {
            let p = first_non_ws(hay, from, limit);
            ns.iter().find_map(|n| one(n, true, p, true))
        }
        Pat::Seq(ns) | Pat::SeqWord(ns) => {
            let word = matches!(pat, Pat::SeqWord(_));
            let mut pos = from;
            for (k, n) in ns.iter().enumerate() {
                pos = one(n, word, pos, anchored && k == 0)?;
            }
            Some(pos)
        }
        Pat::Flag(n) => flag_find(hay, from, limit, n),
        Pat::Lit(..) | Pat::Check(_) => None,
    }
}

fn skip_decor(b: &[u8], from: usize, to: usize) -> usize {
    let mut p = from;
    while p < to && matches!(at(b, p), b' ' | b'\t' | b'*' | b'/' | b'!' | b'-') {
        p += 1;
    }
    p
}

/// Directive positions on a line: (start, segment end) for each comment and
/// for each later segment its own line marker opens (`# a  # noqa`).
pub fn directive_starts(line: &Line<'_>) -> Vec<(usize, usize)> {
    let b = line.raw.as_bytes();
    let mut out = Vec::new();
    for seg in &line.comments {
        out.push((skip_decor(b, seg.body, seg.end), seg.end));
        if let Some(marker) = seg.marker {
            let mut from = seg.body;
            while let Some((_, e)) = find(b, from, seg.end, marker, false, false) {
                out.push((skip_decor(b, e, seg.end), seg.end));
                from = e;
            }
        }
    }
    out
}

/// Phrases that mark a raised error as a placeholder.
const STUB_PHRASES: &[&str] = &[
    "todo",
    "not implemented",
    "unimplemented",
    "not yet implemented",
    "implement me",
    "fixme",
];

fn lit_has(line: &Line<'_>, phrases: &[&str]) -> bool {
    line.lits.iter().any(|&(s, e)| {
        let content = line.raw.get(s..e).unwrap_or("").to_ascii_lowercase();
        phrases.iter().any(|p| content.contains(p))
    })
}

enum Found {
    No,
    At(usize),
    Justified,
}

fn find_rule(rule: &Rule, doc: &Doc<'_>, i: usize) -> Found {
    let Some(line) = doc.lines.get(i) else {
        return Found::No;
    };
    let raw = line.raw.as_bytes();
    let found = match rule.pat {
        Pat::Check(f) => {
            return match f(doc, i) {
                CheckHit::No => Found::No,
                CheckHit::Yes => Found::At(0),
                CheckHit::Justified => Found::Justified,
            }
        }
        Pat::Lit(callee, phrases) => {
            let code = line.code.as_bytes();
            find(code, 0, code.len(), callee, rule.ci, true)
                .filter(|_| lit_has(line, phrases))
                .map(|(_, e)| e)
        }
        ref pat => match rule.site {
            Site::Directive => directive_starts(line)
                .into_iter()
                .find_map(|(p, end)| pat_in(pat, raw, p, end, rule.ci, true)),
            Site::Comment => line
                .comments
                .iter()
                .find_map(|s| pat_in(pat, raw, s.body, s.end, rule.ci, false)),
            Site::Code => pat_in(
                pat,
                line.code.as_bytes(),
                0,
                line.code.len(),
                rule.ci,
                false,
            ),
            Site::Text => pat_in(
                pat,
                line.text.as_bytes(),
                0,
                line.text.len(),
                rule.ci,
                false,
            ),
        },
    };
    found.map_or(Found::No, Found::At)
}

fn has_words(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphanumeric()).count() >= 3
}

/// cheat.rs's waiver policy: `audit-allow:` or the word `reason`, each
/// followed by words.
fn comment_has_reason(comment: &str) -> bool {
    let lower = comment.to_lowercase();
    if let Some(at) = lower.find("audit-allow:") {
        if has_words(lower.get(at + "audit-allow:".len()..).unwrap_or("")) {
            return true;
        }
    }
    let b = lower.as_bytes();
    let mut from = 0;
    while let Some((s, e)) = find(b, from, b.len(), "reason", false, true) {
        if has_words(lower.get(e..).unwrap_or("")) {
            return true;
        }
        from = s + 1;
    }
    false
}

fn reason_near(doc: &Doc<'_>, i: usize) -> bool {
    let here = doc
        .lines
        .get(i)
        .is_some_and(|l| comment_has_reason(&l.comment_text()));
    let above = i
        .checked_sub(1)
        .and_then(|j| doc.lines.get(j))
        .is_some_and(|l| l.is_comment_only() && comment_has_reason(&l.comment_text()));
    here || above
}

/// Two or more words of three or more letters: prose, not a code or a
/// placeholder like `-- ok`.
fn is_prose(text: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.chars().filter(char::is_ascii_alphabetic).count() >= 3)
        .count()
        >= 2
}

fn native_prose(raw: &str, end: usize, sep: &str) -> bool {
    let tail = raw.get(end..).unwrap_or("");
    if sep.is_empty() {
        return is_prose(tail);
    }
    let lower = tail.to_ascii_lowercase();
    lower
        .find(&sep.to_ascii_lowercase())
        .and_then(|k| tail.get(k + sep.len()..))
        .is_some_and(is_prose)
}

fn escalates(esc: &Escalate, doc: &Doc<'_>, i: usize) -> bool {
    let has = |j: usize| {
        doc.lines
            .get(j)
            .is_some_and(|l| esc.markers.iter().any(|m| l.text.contains(m)))
    };
    match esc.scope {
        Scope::File => (0..doc.lines.len()).any(has),
        Scope::Near => has(i) || next_code_line(doc, i).is_some_and(has),
    }
}

fn evidence(raw: &str) -> String {
    let t = raw.trim();
    if t.chars().count() > 120 {
        t.chars().take(117).collect::<String>() + "..."
    } else {
        t.to_string()
    }
}

fn apply(rule: &Rule, doc: &Doc<'_>, i: usize) -> Option<Hit> {
    let line = doc.lines.get(i)?;
    if !rule.requires.is_empty() {
        let text = line.text.as_bytes();
        let present = rule
            .requires
            .iter()
            .any(|w| find(text, 0, text.len(), w, rule.ci, true).is_some());
        if !present {
            return None;
        }
    }
    let raw = line.raw.as_bytes();
    if rule
        .except
        .iter()
        .any(|e| find(raw, 0, raw.len(), e, rule.ci, false).is_some())
    {
        return None;
    }
    let (end, native) = match find_rule(rule, doc, i) {
        Found::No => return None,
        Found::At(end) => (end, false),
        Found::Justified => (0, true),
    };
    let mut severity = rule.severity;
    if let Some(esc) = &rule.escalate {
        if escalates(esc, doc, i) {
            severity = severity.max(esc.to);
        }
    }
    let justified = native
        || match rule.waiver {
            Waiver::Never => false,
            Waiver::Reason => reason_near(doc, i),
            Waiver::Native(sep) => reason_near(doc, i) || native_prose(line.raw, end, sep),
        };
    if justified && rule.waiver != Waiver::Never {
        severity = Severity::Note;
    }
    Some(Hit {
        kind: rule.kind,
        lang: doc.lang,
        severity,
        line: Some(line.no),
        evidence: evidence(line.raw),
        because: rule.because,
        tool: rule.tool,
        rule: rule.id,
    })
}

/// Scan `text` as `lang` with every rule that applies to it.
pub fn scan(text: &str, lang: Lang) -> Vec<Hit> {
    scan_doc(&lex(text, lang))
}

/// [`scan`] for a text whose path is known. A Python stub file (`.pyi`) is
/// declarations only, so its `...` bodies are not stubs presented as
/// implementations and `py.ellipsis-body` does not apply; every other rule
/// runs exactly as in [`scan`].
pub fn scan_path(text: &str, lang: Lang, path: Option<&str>) -> Vec<Hit> {
    let stub_file =
        lang == Lang::Python && path.is_some_and(|p| p.to_ascii_lowercase().ends_with(".pyi"));
    let mut hits = scan(text, lang);
    if stub_file {
        hits.retain(|h| h.rule != "py.ellipsis-body");
    }
    hits
}

/// [`scan`] over an already-lexed text.
pub fn scan_doc(doc: &Doc<'_>) -> Vec<Hit> {
    let mut hits: Vec<Hit> = applicable_rules(doc.lang)
        .flat_map(|rule| (0..doc.lines.len()).filter_map(move |i| apply(rule, doc, i)))
        .collect();
    hits.sort_by(|a, b| a.line.cmp(&b.line).then_with(|| a.rule.cmp(b.rule)));
    hits
}

/// The highest severity among `hits`, or `None` when there are none.
pub fn worst(hits: &[Hit]) -> Option<Severity> {
    hits.iter().map(|h| h.severity).max()
}

/// The language's own rule table. [`scan`] also applies the shared tables
/// [`applicable_rules`] lists (C-family rules for C++, JS rules for TS,
/// shell rules for CI and Makefiles, ...).
pub fn rules(lang: Lang) -> &'static [Rule] {
    tables(lang).first().copied().unwrap_or(&[])
}

/// Every rule [`scan`] applies to `lang`, own table first.
pub fn applicable_rules(lang: Lang) -> impl Iterator<Item = &'static Rule> {
    tables(lang).iter().flat_map(|t| t.iter())
}

/// The catalog entry behind a [`Hit::rule`], e.g. to drop hits whose
/// [`Rule::overlap`] a gate-side scan already reports.
pub fn rule_by_id(id: &str) -> Option<&'static Rule> {
    Lang::ALL
        .iter()
        .flat_map(|&lang| tables(lang).iter())
        .flat_map(|t| t.iter())
        .find(|r| r.id == id)
}

fn tables(lang: Lang) -> &'static [&'static [Rule]] {
    match lang {
        Lang::Rust => &[RUST, UNIVERSAL],
        Lang::Python => &[PYTHON, UNIVERSAL],
        Lang::C => &[C_FAMILY, UNIVERSAL],
        Lang::Cpp => &[CPP, C_FAMILY, UNIVERSAL],
        Lang::ObjectiveC => &[OBJC, C_FAMILY, UNIVERSAL],
        Lang::CSharp => &[CSHARP, UNIVERSAL],
        Lang::Java => &[JAVA, UNIVERSAL],
        Lang::Kotlin => &[KOTLIN, JAVA, UNIVERSAL],
        Lang::Scala => &[SCALA, JAVA, UNIVERSAL],
        Lang::Go => &[GO, UNIVERSAL],
        Lang::JavaScript => &[JAVASCRIPT, UNIVERSAL],
        Lang::TypeScript => &[TYPESCRIPT, JAVASCRIPT, UNIVERSAL],
        Lang::Ruby => &[RUBY, UNIVERSAL],
        Lang::Php => &[PHP, UNIVERSAL],
        Lang::Swift => &[SWIFT, UNIVERSAL],
        Lang::Dart => &[DART, UNIVERSAL],
        Lang::Lua => &[LUA, UNIVERSAL],
        Lang::Perl => &[PERL, UNIVERSAL],
        Lang::R => &[R_LANG, UNIVERSAL],
        Lang::Julia => &[JULIA, UNIVERSAL],
        Lang::Haskell => &[HASKELL, UNIVERSAL],
        Lang::OCaml => &[OCAML, UNIVERSAL],
        Lang::Elixir => &[ELIXIR, UNIVERSAL],
        Lang::Erlang => &[ERLANG, UNIVERSAL],
        Lang::Clojure => &[CLOJURE, UNIVERSAL],
        Lang::Zig => &[ZIG, UNIVERSAL],
        Lang::Nim => &[NIM, UNIVERSAL],
        Lang::Solidity => &[SOLIDITY, UNIVERSAL],
        Lang::Shell => &[SHELL, GATE_FLAGS, COMPILER_FLAGS, UNIVERSAL],
        Lang::PowerShell => &[POWERSHELL, UNIVERSAL],
        Lang::Sql => &[SQL, UNIVERSAL],
        Lang::Terraform => &[TERRAFORM, IAC],
        Lang::Dockerfile => &[DOCKERFILE, IAC, SHELL, GATE_FLAGS, COMPILER_FLAGS],
        Lang::Makefile => &[MAKEFILE, SHELL, GATE_FLAGS, COMPILER_FLAGS],
        Lang::Yaml => &[YAML, IAC],
        Lang::CiConfig => &[CI, YAML, SHELL, GATE_FLAGS, COMPILER_FLAGS],
        Lang::CargoManifest => &[CARGO],
        Lang::ClippyConfig => &[CLIPPY],
        Lang::PythonConfig => &[PYTHON_CONFIG, GATE_FLAGS],
        Lang::JsLintConfig => &[JS_LINT_CONFIG],
        Lang::TsConfig => &[TS_CONFIG],
        Lang::NodePackage => &[NODE_PACKAGE, SHELL, GATE_FLAGS],
        Lang::GolangciConfig => &[GOLANGCI],
        Lang::MsBuild => &[MSBUILD],
        Lang::EditorConfig => &[EDITORCONFIG],
        Lang::Gradle => &[GRADLE],
        Lang::Maven => &[MAVEN],
        Lang::RubocopConfig => &[RUBOCOP_CONFIG],
        Lang::PhpLintConfig => &[PHP_LINT_CONFIG],
        Lang::SwiftlintConfig => &[SWIFTLINT_CONFIG],
        Lang::DartAnalysis => &[DART_ANALYSIS],
        Lang::CMake => &[CMAKE, COMPILER_FLAGS],
        Lang::ClangTidyConfig => &[CLANG_TIDY_CONFIG],
    }
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

const LS: PolyKind = PolyKind::LintSuppression;
const SKIP: PolyKind = PolyKind::TestSkip;
const WEAK: PolyKind = PolyKind::TestWeakening;
const STUB: PolyKind = PolyKind::Stub;
const CFG: PolyKind = PolyKind::ConfigWeakening;
const TYPE: PolyKind = PolyKind::TypeCheckBypass;
const ESC: PolyKind = PolyKind::UnsafeEscape;
const NOTE: Severity = Severity::Note;
const EVA: Severity = Severity::Evasion;
const SUP: Severity = Severity::Suppression;
const FRAUD: Severity = Severity::Fraud;

const fn rule(
    id: &'static str,
    kind: PolyKind,
    severity: Severity,
    site: Site,
    pat: Pat,
    because: &'static str,
) -> Rule {
    let waiver = match kind {
        PolyKind::LintSuppression
        | PolyKind::TypeCheckBypass
        | PolyKind::ConfigWeakening
        | PolyKind::UnsafeEscape => Waiver::Reason,
        PolyKind::TestSkip | PolyKind::TestWeakening | PolyKind::Stub => Waiver::Never,
    };
    Rule {
        id,
        kind,
        severity,
        tool: None,
        site,
        pat,
        ci: false,
        waiver,
        requires: &[],
        except: &[],
        escalate: None,
        overlap: None,
        because,
    }
}

/// A directive: `needle` opening a comment.
const fn dir(
    id: &'static str,
    kind: PolyKind,
    severity: Severity,
    tool: &'static str,
    pat: Pat,
    because: &'static str,
) -> Rule {
    rule(id, kind, severity, Site::Directive, pat, because).tool(tool)
}

const fn code(
    id: &'static str,
    kind: PolyKind,
    severity: Severity,
    pat: Pat,
    because: &'static str,
) -> Rule {
    rule(id, kind, severity, Site::Code, pat, because)
}

const fn text(
    id: &'static str,
    kind: PolyKind,
    severity: Severity,
    pat: Pat,
    because: &'static str,
) -> Rule {
    rule(id, kind, severity, Site::Text, pat, because)
}

const fn check(
    id: &'static str,
    kind: PolyKind,
    severity: Severity,
    f: CheckFn,
    because: &'static str,
) -> Rule {
    rule(id, kind, severity, Site::Code, Pat::Check(f), because)
}

/// Markers that put a coverage exclusion or build tag on test code.
const TEST_MARKERS: &[&str] = &[
    "#[test]",
    "def test",
    "it(",
    "test(",
    "describe(",
    "func Test",
    "@Test",
    "assert",
    "expect(",
    "TEST(",
    "TEST_F(",
    "[Fact]",
    "[Test]",
    "[TestMethod]",
    "XCTAssert",
];

/// Commands whose failure is the gate: `|| true` after one of these hides it.
const GATE_CMDS: &[&str] = &[
    "pytest",
    "py.test",
    "tox",
    "nox",
    "unittest",
    "cargo",
    "rustc",
    "clippy-driver",
    "go",
    "golangci-lint",
    "staticcheck",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "bun",
    "deno",
    "jest",
    "vitest",
    "mocha",
    "eslint",
    "tsc",
    "biome",
    "ruff",
    "mypy",
    "pyright",
    "flake8",
    "pylint",
    "bandit",
    "make",
    "ctest",
    "mvn",
    "gradle",
    "gradlew",
    "dotnet",
    "rspec",
    "rake",
    "rubocop",
    "phpunit",
    "phpstan",
    "psalm",
    "phpcs",
    "pest",
    "swift",
    "swiftlint",
    "dart",
    "flutter",
    "mix",
    "rebar3",
    "cabal",
    "stack",
    "dune",
    "zig",
    "nimble",
    "forge",
    "hardhat",
    "shellcheck",
    "hadolint",
    "yamllint",
    "sqlfluff",
    "tflint",
    "checkov",
    "Invoke-Pester",
    "audit-bad-patterns.sh",
];

/// Compiler invocations and the variables that feed them.
const COMPILERS: &[&str] = &[
    "gcc",
    "g++",
    "clang",
    "clang++",
    "cc",
    "c++",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "CMAKE_C_FLAGS",
    "CMAKE_CXX_FLAGS",
    "add_compile_options",
    "target_compile_options",
    "COMPILE_OPTIONS",
];

const RUST_TOOLS: &[&str] = &["clippy", "rustc", "RUSTFLAGS", "cargo", "clippy-driver"];

/// Directives every code language's comments can carry: cross-language
/// scanners and IDE suppressions.
const UNIVERSAL: &[Rule] = &[
    rule(
        "any.nosonar",
        LS,
        SUP,
        Site::Comment,
        Pat::Has("NOSONAR"),
        "a `NOSONAR` comment silences SonarQube on the line instead of fixing the issue",
    )
    .tool("sonarqube"),
    rule(
        "any.nosemgrep",
        LS,
        SUP,
        Site::Comment,
        Pat::WordAny(&["nosemgrep", "nosem"]),
        "a `nosemgrep` comment silences the security scanner on the line",
    )
    .tool("semgrep")
    .ci(),
    rule(
        "any.coverity",
        LS,
        SUP,
        Site::Comment,
        Pat::Has("coverity["),
        "a `coverity[...]` annotation suppresses a static-analysis defect",
    )
    .tool("coverity"),
    rule(
        "any.codeql",
        LS,
        SUP,
        Site::Comment,
        Pat::HasAny(&["codeql[", "lgtm["]),
        "a `codeql[...]`/`lgtm[...]` comment suppresses a code-scanning alert",
    )
    .tool("codeql"),
    dir(
        "any.deepcode",
        LS,
        SUP,
        "snyk",
        Pat::Has("deepcode ignore"),
        "a `deepcode ignore` comment suppresses a Snyk Code finding",
    )
    .native(":"),
    dir(
        "any.noinspection",
        LS,
        SUP,
        "jetbrains",
        Pat::Word("noinspection"),
        "a `noinspection` comment silences the IDE inspection instead of fixing the code",
    ),
    dir(
        "any.formatter-off",
        LS,
        NOTE,
        "jetbrains",
        Pat::Has("@formatter:off"),
        "formatting is switched off for a region; recorded, not a defect by itself",
    ),
];

const RUST: &[Rule] = &[
    check("rust.cfg-false", SKIP, SUP, rust_always_false_cfg, "an always-false `cfg` compiles the item out: a test that is never built never fails").overlap("cheat::scan (gate-side is_always_false_cfg)"),
    check("rust.cfg-attr-ignore", SKIP, SUP, rust_cfg_attr_ignore, "a `cfg_attr(.., ignore)` switches a test off under a condition the gate meets").overlap("cheat::scan (gate-side is_ignore_attr)"),
    check("rust.should-panic-bare", WEAK, SUP, rust_should_panic_bare, "a bare `#[should_panic]` passes on any panic, including the wrong one").overlap("cheat::scan (gate-side)"),
    check("rust.lint-attr-long", LS, SUP, rust_lint_attr_long, "a lint attribute spread over many lines silences the check where a line-window scan cannot see it"),
    code("rust.panic-stub", STUB, EVA, Pat::Lit("panic!", STUB_PHRASES), "a `panic!` whose message says not implemented is a hole presented as an implementation"),
    code("rust.unreachable-stub", STUB, EVA, Pat::Lit("unreachable!", STUB_PHRASES), "an `unreachable!` whose message says todo is a hole presented as an implementation"),
    code("rust.rustfmt-skip", LS, NOTE, Pat::Has("rustfmt::skip"), "formatting is switched off for an item; recorded, not a defect by itself").tool("rustfmt"),
    code("rust.coverage-off", WEAK, NOTE, Pat::Has("coverage(off)"), "the item is excluded from coverage measurement").tool("llvm-cov").near(TEST_MARKERS, EVA),
    check("rust.let-underscore", ESC, EVA, rust_let_underscore_io, "`let _ = ` on a write or a filesystem change discards the error that says it did not happen").tool("clippy"),
    check("rust.ok-discard", ESC, EVA, rust_ok_discard, "a statement `.ok();` on a write or a filesystem change turns the error that says it did not happen into nothing").tool("clippy"),
];

const PYTHON: &[Rule] = &[
    dir("py.pyright-basic", TYPE, SUP, "pyright", Pat::Has("pyright: basic"), "`# pyright: basic` lowers the file's type-checking level below the project's"),
    dir("py.pyright-report-false", TYPE, SUP, "pyright", Pat::Seq(&["pyright:", "report", "= false"]), "`# pyright: reportX=false` switches a check off for the file"),
    dir("py.pyright-report-none", TYPE, SUP, "pyright", Pat::Seq(&["pyright:", "report", "= none"]), "`# pyright: reportX=none` switches a check off for the file"),
    dir("py.mypy-ignore-errors", TYPE, SUP, "mypy", Pat::Has("mypy: ignore-errors"), "`# mypy: ignore-errors` turns the type checker off for the whole file"),
    dir("py.mypy-disable-code", TYPE, SUP, "mypy", Pat::Has("mypy: disable-error-code"), "`# mypy: disable-error-code` silences an error class for the whole file"),
    dir("py.pyre-ignore", TYPE, SUP, "pyre", Pat::HasAny(&["pyre-ignore", "pyre-fixme"]), "a `pyre-ignore`/`pyre-fixme` comment silences the type checker"),
    dir("py.pyre-unsafe", TYPE, EVA, "pyre", Pat::Has("pyre-unsafe"), "`# pyre-unsafe` lowers the file's checking mode"),
    dir("py.pytype-disable", TYPE, SUP, "pytype", Pat::Has("pytype: disable"), "a `pytype: disable` comment silences the type checker"),
    dir("py.nosec", LS, SUP, "bandit", Pat::Word("nosec"), "a `# nosec` comment silences the security linter on the line").ci(),
    dir("py.no-cover", WEAK, NOTE, "coverage.py", Pat::HasAny(&["pragma: no cover", "pragma no cover"]), "the code is excluded from coverage measurement").ci().near(TEST_MARKERS, EVA),
    dir("py.fmt-off", LS, NOTE, "black", Pat::HasAny(&["fmt: off", "fmt: skip"]), "formatting is switched off; recorded, not a defect by itself"),
    dir("py.isort-skip", LS, NOTE, "isort", Pat::Has("isort: skip"), "import sorting is switched off; recorded, not a defect by itself"),
    dir("py.pylint-skip-file", LS, SUP, "pylint", Pat::Has("pylint: skip-file"), "`# pylint: skip-file` turns the linter off for the whole file"),
    code("py.skipif", SKIP, EVA, Pat::Has("@pytest.mark.skipif"), "a conditional skip whose condition is not a platform or version gate can switch the test off where the gate runs").tool("pytest").except(&["sys.platform", "sys.version_info", "platform.", "os.name", "find_spec"]),
    code("py.unittest-skipif", SKIP, EVA, Pat::HasAny(&["@unittest.skipIf", "@unittest.skipUnless"]), "a conditional skip whose condition is not a platform or version gate can switch the test off where the gate runs").tool("unittest").except(&["sys.platform", "sys.version_info", "platform.", "os.name", "find_spec"]),
    code("py.xfail", WEAK, SUP, Pat::Has("pytest.mark.xfail"), "a non-strict `xfail` lets a failing test pass, and a fixed one pass silently too").tool("pytest").except(&["strict=True"]),
    code("py.xfail-call", WEAK, SUP, Pat::Word("pytest.xfail("), "`pytest.xfail()` turns a running test into an expected failure").tool("pytest"),
    code("py.expected-failure", WEAK, SUP, Pat::Has("@unittest.expectedFailure"), "`@expectedFailure` makes the test's failure its passing state").tool("unittest"),
    code("py.skiptest-raise", SKIP, SUP, Pat::Word("SkipTest"), "raising `SkipTest` switches the test off from inside").tool("unittest"),
    check("py.ellipsis-body", STUB, EVA, py_ellipsis_body, "a `...` body behind a `def` is an empty function presented as an implementation"),
    code("py.cast-any", TYPE, EVA, Pat::Has("cast(Any"), "`cast(Any, x)` tells the type checker to stop checking the value").tool("mypy"),
    code("py.suppress-exception", ESC, EVA, Pat::WordAny(&["suppress(Exception)", "suppress(BaseException)"]), "`contextlib.suppress(Exception)` swallows every error in the block"),
    text("py.filterwarnings-ignore", ESC, EVA, Pat::SeqWord(&["warnings", "filterwarnings", "ignore"]), "ignoring all warnings at runtime hides deprecations and resource leaks"),
    text("py.simplefilter-ignore", ESC, EVA, Pat::SeqWord(&["simplefilter", "ignore"]), "ignoring all warnings at runtime hides deprecations and resource leaks"),
];

const C_FAMILY: &[Rule] = &[
    dir(
        "c.cppcheck-suppress",
        LS,
        SUP,
        "cppcheck",
        Pat::Has("cppcheck-suppress"),
        "a `cppcheck-suppress` comment silences the analyzer on the line",
    ),
    code(
        "c.pragma-warning-level0",
        LS,
        SUP,
        Pat::Has("#pragma warning ( push , 0 )"),
        "`#pragma warning(push, 0)` drops the warning level to zero for everything after it",
    )
    .tool("msvc"),
    code(
        "c.system-header",
        LS,
        SUP,
        Pat::HasAny(&["#pragma GCC system_header", "#pragma clang system_header"]),
        "`#pragma system_header` silences every warning in the rest of the file",
    )
    .tool("gcc"),
    code(
        "c.diag-suppress",
        LS,
        SUP,
        Pat::Seq(&["#pragma", "diag_suppress"]),
        "`#pragma diag_suppress` silences a compiler diagnostic",
    )
    .tool("nvcc"),
    code(
        "c.if-zero",
        SKIP,
        EVA,
        Pat::Word("#if 0"),
        "`#if 0` compiles a block out: whatever it holds is never built or run",
    ),
    code(
        "c.gtest-skip",
        SKIP,
        SUP,
        Pat::Word("GTEST_SKIP"),
        "`GTEST_SKIP()` switches the test off from inside",
    )
    .tool("googletest"),
    code(
        "c.gtest-disabled",
        SKIP,
        SUP,
        Pat::Seq(&["TEST", "(", "DISABLED_"]),
        "a `DISABLED_` test name tells googletest never to run it",
    )
    .tool("googletest"),
    code(
        "c.ndebug",
        WEAK,
        SUP,
        Pat::Has("#define NDEBUG"),
        "defining `NDEBUG` in source compiles every `assert` out",
    ),
    rule(
        "c.lcov-excl",
        WEAK,
        NOTE,
        Site::Comment,
        Pat::HasAny(&["LCOV_EXCL", "GCOVR_EXCL"]),
        "the code is excluded from coverage measurement",
    )
    .tool("lcov")
    .near(TEST_MARKERS, EVA),
    text(
        "c.catch2-hidden",
        SKIP,
        SUP,
        Pat::Seq(&["TEST_CASE", "[."]),
        "a `[.]`-tagged Catch2 test is hidden from the default run",
    )
    .tool("catch2"),
    text(
        "c.catch2-mayfail",
        WEAK,
        SUP,
        Pat::SeqWord(&["TEST_CASE", "[!mayfail]"]),
        "a `[!mayfail]` Catch2 test cannot fail the run",
    )
    .tool("catch2"),
    text(
        "c.catch2-shouldfail",
        WEAK,
        SUP,
        Pat::Seq(&["TEST_CASE", "[!shouldfail]"]),
        "a `[!shouldfail]` Catch2 test passes by failing",
    )
    .tool("catch2"),
    code(
        "c.doctest-skip",
        SKIP,
        SUP,
        Pat::Word("doctest::skip"),
        "a `doctest::skip()` decorator switches the test off",
    )
    .tool("doctest"),
    code(
        "c.boost-disabled",
        SKIP,
        SUP,
        Pat::Has("unit_test::disabled()"),
        "a `disabled()` decorator switches the Boost.Test case off",
    )
    .tool("boost.test"),
    dir(
        "c.commented-test",
        SKIP,
        EVA,
        "googletest",
        Pat::WordAny(&["TEST(", "TEST_F(", "TEST_P(", "TEST_CASE("]),
        "a commented-out test is presented as coverage it no longer provides",
    ),
    code(
        "c.assert-stub",
        STUB,
        EVA,
        Pat::Lit("assert", STUB_PHRASES),
        "`assert(0 && \"not implemented\")` is a hole presented as an implementation",
    ),
];

const CPP: &[Rule] = &[
    code(
        "cpp.throw-stub",
        STUB,
        EVA,
        Pat::Lit("throw", STUB_PHRASES),
        "a `throw ...(\"not implemented\")` is a hole presented as an implementation",
    ),
    check(
        "cpp.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
];

const OBJC: &[Rule] = &[
    code(
        "objc.xctskip",
        SKIP,
        SUP,
        Pat::Word("XCTSkip("),
        "`XCTSkip` switches the test off from inside",
    )
    .tool("xctest"),
    code(
        "objc.xctskip-if",
        SKIP,
        EVA,
        Pat::WordAny(&["XCTSkipIf(", "XCTSkipUnless("]),
        "a conditional `XCTSkip` can switch the test off where the gate runs",
    )
    .tool("xctest"),
    code(
        "objc.block-assertions",
        WEAK,
        SUP,
        Pat::Word("NS_BLOCK_ASSERTIONS"),
        "`NS_BLOCK_ASSERTIONS` compiles every `NSAssert` out",
    ),
    code(
        "objc.throw-stub",
        STUB,
        EVA,
        Pat::Lit("@throw", STUB_PHRASES),
        "an `@throw` with a not-implemented reason is a hole presented as an implementation",
    ),
];

const CSHARP: &[Rule] = &[
    code(
        "cs.suppress-message",
        LS,
        SUP,
        Pat::WordAny(&["SuppressMessage", "SuppressMessageAttribute"]),
        "`[SuppressMessage]` silences an analyzer instead of fixing the code",
    )
    .tool("roslyn")
    .native("Justification"),
    code(
        "cs.exclude-coverage",
        WEAK,
        NOTE,
        Pat::Word("ExcludeFromCodeCoverage"),
        "the code is excluded from coverage measurement",
    )
    .near(TEST_MARKERS, EVA),
    code(
        "cs.xunit-skip",
        SKIP,
        SUP,
        Pat::Has("Skip ="),
        "`Skip = ...` on a test attribute switches the test off",
    )
    .tool("xunit")
    .requires(&["Fact", "Theory", "SkippableFact", "SkippableTheory"]),
    code(
        "cs.ignore-attr",
        SKIP,
        SUP,
        Pat::WordAny(&["[Ignore", "[Explicit"]),
        "`[Ignore]`/`[Explicit]` keeps the test out of the normal run",
    )
    .tool("nunit"),
    code(
        "cs.inconclusive",
        SKIP,
        SUP,
        Pat::WordAny(&[
            "Assert.Inconclusive(",
            "Assert.Ignore(",
            "Assume.That(false",
        ]),
        "an inconclusive/ignored assertion turns the test into a skip",
    )
    .tool("mstest"),
    code(
        "cs.skippable",
        SKIP,
        EVA,
        Pat::WordAny(&["Skip.If(", "Skip.IfNot("]),
        "a conditional skip can switch the test off where the gate runs",
    )
    .tool("xunit"),
    code(
        "cs.nullable-disable",
        TYPE,
        SUP,
        Pat::Word("#nullable disable"),
        "`#nullable disable` turns null-safety analysis off for the rest of the file",
    )
    .tool("roslyn"),
    code(
        "cs.null-forgiving",
        TYPE,
        EVA,
        Pat::Word("null!"),
        "`null!` tells the compiler a null is not null",
    )
    .tool("roslyn"),
    code(
        "cs.not-implemented",
        STUB,
        EVA,
        Pat::Word("NotImplementedException"),
        "`throw new NotImplementedException()` is a hole presented as an implementation",
    ),
    dir(
        "cs.resharper-disable",
        LS,
        SUP,
        "resharper",
        Pat::Has("ReSharper disable"),
        "a `ReSharper disable` comment silences the inspection",
    ),
    code(
        "cs.if-false",
        SKIP,
        EVA,
        Pat::Word("#if false"),
        "`#if false` compiles a block out: whatever it holds is never built or run",
    ),
    check(
        "cs.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
    dir(
        "cs.commented-test",
        SKIP,
        EVA,
        "xunit",
        Pat::HasAny(&["[Fact]", "[Test]", "[TestMethod]", "[Theory]"]),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const JAVA: &[Rule] = &[
    code(
        "java.suppress-warnings",
        LS,
        SUP,
        Pat::Word("@SuppressWarnings"),
        "`@SuppressWarnings` silences the compiler or a linter instead of fixing the code",
    )
    .tool("javac"),
    code(
        "java.suppress-lint",
        LS,
        SUP,
        Pat::Word("@SuppressLint"),
        "`@SuppressLint` silences Android lint",
    )
    .tool("android-lint"),
    code(
        "java.suppress-fb",
        LS,
        SUP,
        Pat::Word("@SuppressFBWarnings"),
        "`@SuppressFBWarnings` silences SpotBugs",
    )
    .tool("spotbugs")
    .native("justification"),
    rule(
        "java.nopmd",
        LS,
        SUP,
        Site::Comment,
        Pat::Word("NOPMD"),
        "a `NOPMD` comment silences PMD on the line",
    )
    .tool("pmd")
    .native(""),
    rule(
        "java.checkstyle-off",
        LS,
        SUP,
        Site::Comment,
        Pat::WordAny(&[
            "CHECKSTYLE:OFF",
            "CHECKSTYLE.OFF",
            "CHECKSTYLE IGNORE",
            "SUPPRESS CHECKSTYLE",
        ]),
        "a Checkstyle suppression comment switches the check off",
    )
    .tool("checkstyle"),
    code(
        "java.disabled",
        SKIP,
        SUP,
        Pat::WordAny(&["@Disabled", "@Ignore"]),
        "`@Disabled`/`@Ignore` keeps the test from running",
    )
    .tool("junit"),
    code(
        "java.testng-disabled",
        SKIP,
        SUP,
        Pat::Seq(&["@Test", "enabled = false"]),
        "`@Test(enabled = false)` keeps the test from running",
    )
    .tool("testng"),
    code(
        "java.assume-false",
        SKIP,
        SUP,
        Pat::WordAny(&[
            "assumeTrue ( false )",
            "assumeFalse ( true )",
            "Assumptions.abort (",
        ]),
        "an assumption that cannot hold aborts the test as skipped",
    )
    .tool("junit"),
    code(
        "java.vacuous-assert",
        WEAK,
        SUP,
        Pat::WordAny(&["assertTrue ( true )", "assertFalse ( false )"]),
        "an assertion on a constant cannot fail",
    )
    .tool("junit"),
    code(
        "java.throw-stub",
        STUB,
        EVA,
        Pat::Lit("throw new", STUB_PHRASES),
        "a `throw new ...(\"not implemented\")` is a hole presented as an implementation",
    ),
    rule(
        "java.auto-stub",
        STUB,
        EVA,
        Site::Comment,
        Pat::Has("Auto-generated method stub"),
        "an IDE-generated stub marker says the body was never written",
    )
    .ci(),
    check(
        "java.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
    dir(
        "java.commented-test",
        SKIP,
        EVA,
        "junit",
        Pat::Word("@Test"),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const KOTLIN: &[Rule] = &[
    code(
        "kt.suppress",
        LS,
        SUP,
        Pat::WordAny(&["@Suppress", "@file:Suppress"]),
        "`@Suppress` silences the compiler or detekt instead of fixing the code",
    )
    .tool("kotlinc"),
    dir(
        "kt.ktlint-disable",
        LS,
        SUP,
        "ktlint",
        Pat::Has("ktlint-disable"),
        "a `ktlint-disable` comment switches the linter off",
    ),
    code(
        "kt.todo",
        STUB,
        EVA,
        Pat::Word("TODO("),
        "Kotlin's `TODO()` throws: a hole presented as an implementation",
    ),
];

const SCALA: &[Rule] = &[
    code(
        "scala.nowarn",
        LS,
        SUP,
        Pat::Word("@nowarn"),
        "`@nowarn` silences compiler warnings",
    )
    .tool("scalac"),
    dir(
        "scala.scalafix-off",
        LS,
        SUP,
        "scalafix",
        Pat::WordAny(&["scalafix:off", "scalafix:ok"]),
        "a `scalafix:off`/`ok` comment switches the linter off",
    ),
    dir(
        "scala.scalastyle-off",
        LS,
        SUP,
        "scalastyle",
        Pat::WordAny(&["scalastyle:off", "scalastyle:ignore"]),
        "a `scalastyle:off` comment switches the linter off",
    ),
    code(
        "scala.qqq",
        STUB,
        EVA,
        Pat::Has("???"),
        "`???` throws `NotImplementedError`: a hole presented as an implementation",
    ),
    code(
        "scala.ignore",
        SKIP,
        SUP,
        Pat::Lead(&["ignore("]),
        "ScalaTest `ignore(...)` registers the test without running it",
    )
    .tool("scalatest"),
    code(
        "scala.pending",
        SKIP,
        SUP,
        Pat::Lead(&["pending"]),
        "`pending` marks the test as not yet written",
    )
    .tool("scalatest"),
    code(
        "scala.in-pending",
        SKIP,
        SUP,
        Pat::WordAny(&["in pending", "pendingUntilFixed"]),
        "`in pending` registers the test without running it",
    )
    .tool("scalatest"),
    code(
        "scala.assume-false",
        SKIP,
        SUP,
        Pat::Has("assume(false"),
        "an assumption that cannot hold cancels the test",
    )
    .tool("scalatest"),
    code(
        "scala.munit-ignore",
        SKIP,
        SUP,
        Pat::Has("\".ignore"),
        "munit `.ignore` keeps the test from running",
    )
    .tool("munit"),
];

const GO: &[Rule] = &[
    dir(
        "go.nolint",
        LS,
        SUP,
        "golangci-lint",
        Pat::Word("nolint"),
        "a `//nolint` comment silences the linter on the line",
    )
    .native("//"),
    dir(
        "go.lint-ignore",
        LS,
        SUP,
        "staticcheck",
        Pat::WordAny(&["lint:ignore", "lint:file-ignore"]),
        "a `//lint:ignore` comment silences staticcheck",
    )
    .native(""),
    rule(
        "go.nosec",
        LS,
        SUP,
        Site::Comment,
        Pat::Has("#nosec"),
        "a `#nosec` comment silences the security scanner",
    )
    .tool("gosec")
    .native("--"),
    dir(
        "go.gosec-disable",
        LS,
        SUP,
        "gosec",
        Pat::Has("gosec:disable"),
        "a `gosec:disable` comment silences the security scanner",
    ),
    code(
        "go.skip",
        SKIP,
        SUP,
        Pat::WordAny(&[
            "t.Skip(",
            "t.Skipf(",
            "t.SkipNow(",
            "b.Skip(",
            "b.Skipf(",
            "b.SkipNow(",
            "tb.Skip(",
            "tb.Skipf(",
            "tb.SkipNow(",
        ]),
        "`t.Skip` switches the test off from inside",
    )
    .tool("go test"),
    code(
        "go.discard-err",
        ESC,
        EVA,
        Pat::Word("_ = err"),
        "`_ = err` discards the error the line before produced",
    ),
    code(
        "go.empty-if-err",
        ESC,
        EVA,
        Pat::Has("err != nil { }"),
        "an empty `if err != nil {}` swallows the error",
    ),
    dir(
        "go.build-ignore",
        SKIP,
        NOTE,
        "go",
        Pat::WordAny(&["go:build ignore", "+build ignore"]),
        "`//go:build ignore` excludes the file from every build",
    )
    .in_file(&["testing.T", "testing.B", "testing.M", "testing.F"], SUP),
    dir(
        "go.linkname",
        ESC,
        EVA,
        "go",
        Pat::Has("go:linkname"),
        "`//go:linkname` reaches into another package's unexported symbols",
    ),
    code(
        "go.panic-stub",
        STUB,
        EVA,
        Pat::Lit("panic", STUB_PHRASES),
        "a `panic(\"not implemented\")` is a hole presented as an implementation",
    ),
    dir(
        "go.commented-test",
        SKIP,
        EVA,
        "go test",
        Pat::Has("func Test"),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const JAVASCRIPT: &[Rule] = &[
    check(
        "js.eslint-inline-off",
        LS,
        SUP,
        js_eslint_inline_off,
        "an inline `/* eslint rule: \"off\" */` config comment switches the rule off",
    )
    .tool("eslint"),
    dir(
        "js.biome-ignore",
        LS,
        SUP,
        "biome",
        Pat::Has("biome-ignore"),
        "a `biome-ignore` comment silences the linter",
    )
    .native(":"),
    dir(
        "js.prettier-ignore",
        LS,
        NOTE,
        "prettier",
        Pat::Has("prettier-ignore"),
        "formatting is switched off for a node; recorded, not a defect by itself",
    ),
    dir(
        "js.deno-lint-ignore",
        LS,
        SUP,
        "deno",
        Pat::Has("deno-lint-ignore"),
        "a `deno-lint-ignore` comment silences the linter",
    ),
    dir(
        "js.oxlint-disable",
        LS,
        SUP,
        "oxlint",
        Pat::Has("oxlint-disable"),
        "an `oxlint-disable` comment silences the linter",
    )
    .native("--"),
    dir(
        "js.jshint",
        LS,
        SUP,
        "jshint",
        Pat::HasAny(&["jshint ignore", "jshint -W"]),
        "a JSHint ignore comment silences the linter",
    ),
    dir(
        "js.flow",
        TYPE,
        SUP,
        "flow",
        Pat::WordAny(&["$FlowFixMe", "$FlowIgnore", "$FlowExpectedError", "@noflow"]),
        "a Flow suppression comment silences the type checker",
    ),
    dir(
        "js.coverage-ignore",
        WEAK,
        NOTE,
        "istanbul",
        Pat::HasAny(&["istanbul ignore", "c8 ignore", "v8 ignore"]),
        "the code is excluded from coverage measurement",
    )
    .near(TEST_MARKERS, EVA),
    code(
        "js.only",
        WEAK,
        SUP,
        Pat::WordAny(&[
            "it.only(",
            "describe.only(",
            "test.only(",
            "context.only(",
            "suite.only(",
        ]),
        "`.only` focuses one test and silently skips every other one",
    )
    .tool("jest"),
    code(
        "js.focus",
        WEAK,
        SUP,
        Pat::Lead(&["fit(", "fdescribe("]),
        "`fit`/`fdescribe` focuses one test and silently skips every other one",
    )
    .tool("jasmine"),
    code(
        "js.todo",
        SKIP,
        SUP,
        Pat::WordAny(&["it.todo(", "test.todo("]),
        "a `.todo` test is a placeholder counted as a test",
    )
    .tool("jest"),
    code(
        "js.failing",
        WEAK,
        SUP,
        Pat::WordAny(&["it.failing(", "test.failing(", "test.fails(", "it.fails("]),
        "a `.failing` test passes by failing",
    )
    .tool("jest"),
    code(
        "js.skip-if",
        SKIP,
        EVA,
        Pat::WordAny(&[
            "it.skipIf(",
            "test.skipIf(",
            "describe.skipIf(",
            "it.runIf(",
            "test.runIf(",
        ]),
        "a conditional skip can switch the test off where the gate runs",
    )
    .tool("vitest"),
    code(
        "js.this-skip",
        SKIP,
        SUP,
        Pat::Has("this.skip ( )"),
        "`this.skip()` switches the test off from inside",
    )
    .tool("mocha"),
    code(
        "js.pending",
        SKIP,
        SUP,
        Pat::Lead(&["pending("]),
        "`pending()` switches the test off",
    )
    .tool("jasmine"),
    check(
        "js.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
    code(
        "js.noop-promise-catch",
        ESC,
        EVA,
        Pat::HasAny(&[
            ".catch ( ( ) => { } )",
            ".catch ( ( ) => null )",
            ".catch ( ( ) => undefined )",
            ".catch ( _ => { } )",
            ".catch ( ( _ ) => { } )",
            ".catch ( e => { } )",
            ".catch ( ( e ) => { } )",
            ".catch ( function ( ) { } )",
        ]),
        "a no-op `.catch` swallows the rejection",
    ),
];

const TYPESCRIPT: &[Rule] = &[
    code(
        "ts.as-any",
        TYPE,
        EVA,
        Pat::Word("as any"),
        "`as any` switches type checking off for the value",
    )
    .tool("tsc"),
    code(
        "ts.as-unknown-as",
        TYPE,
        EVA,
        Pat::Word("as unknown as"),
        "`as unknown as T` forces a type the checker rejected",
    )
    .tool("tsc"),
];

const RUBY: &[Rule] = &[
    dir(
        "rb.rubocop-disable",
        LS,
        SUP,
        "rubocop",
        Pat::HasAny(&["rubocop : disable", "rubocop : todo"]),
        "a `rubocop:disable` comment silences the linter",
    )
    .native("--"),
    dir(
        "rb.standard-disable",
        LS,
        SUP,
        "standardrb",
        Pat::Has("standard:disable"),
        "a `standard:disable` comment silences the linter",
    ),
    code(
        "rb.skip",
        SKIP,
        SUP,
        Pat::Lead(&["skip"]),
        "`skip` switches the test off from inside",
    )
    .tool("minitest")
    .except(&["skip:"]),
    code(
        "rb.pending",
        SKIP,
        SUP,
        Pat::Lead(&["pending"]),
        "`pending` marks the test as expected to fail",
    )
    .tool("rspec")
    .except(&["pending:"]),
    code(
        "rb.xit",
        SKIP,
        SUP,
        Pat::Lead(&[
            "xit",
            "xdescribe",
            "xcontext",
            "xspecify",
            "xexample",
            "xscenario",
            "xfeature",
        ]),
        "an `x`-prefixed example is skipped",
    )
    .tool("rspec"),
    text(
        "rb.skip-meta",
        SKIP,
        SUP,
        Pat::WordAny(&[", :skip", "skip: true", "skip: \"", "skip: '"]),
        "`skip:` metadata keeps the example from running",
    )
    .tool("rspec"),
    code(
        "rb.focus",
        WEAK,
        SUP,
        Pat::Lead(&["fit", "fdescribe", "fcontext"]),
        "a focused example silently skips every other one",
    )
    .tool("rspec"),
    text(
        "rb.focus-meta",
        WEAK,
        SUP,
        Pat::Has("focus: true"),
        "focus metadata silently skips every other example",
    )
    .tool("rspec"),
    code(
        "rb.not-implemented",
        STUB,
        EVA,
        Pat::Word("NotImplementedError"),
        "`raise NotImplementedError` is a hole presented as an implementation",
    ),
    dir(
        "rb.nocov",
        WEAK,
        NOTE,
        "simplecov",
        Pat::Has(":nocov:"),
        "the code is excluded from coverage measurement",
    )
    .near(TEST_MARKERS, EVA),
    code(
        "rb.rescue-nil",
        ESC,
        EVA,
        Pat::Word("rescue nil"),
        "`rescue nil` turns every error into nil",
    ),
    code(
        "rb.verbose-nil",
        ESC,
        EVA,
        Pat::Has("$VERBOSE = nil"),
        "`$VERBOSE = nil` silences every interpreter warning",
    ),
    dir(
        "rb.sorbet-ignore",
        TYPE,
        SUP,
        "sorbet",
        Pat::Has("typed: ignore"),
        "`# typed: ignore` turns the type checker off for the file",
    ),
    dir(
        "rb.sorbet-false",
        TYPE,
        EVA,
        "sorbet",
        Pat::Has("typed: false"),
        "`# typed: false` lowers the file's checking level",
    ),
    code(
        "rb.t-unsafe",
        TYPE,
        EVA,
        Pat::Word("T.unsafe("),
        "`T.unsafe` tells Sorbet to stop checking the value",
    )
    .tool("sorbet"),
    dir(
        "rb.steep-ignore",
        TYPE,
        SUP,
        "steep",
        Pat::Has("steep:ignore"),
        "a `steep:ignore` comment silences the type checker",
    ),
    code(
        "rb.assert-true",
        WEAK,
        SUP,
        Pat::Lead(&["assert true"]),
        "an assertion on a constant cannot fail",
    )
    .tool("minitest"),
];

const PHP: &[Rule] = &[
    dir(
        "php.phpstan-ignore",
        LS,
        SUP,
        "phpstan",
        Pat::Has("@phpstan-ignore"),
        "a `@phpstan-ignore` comment silences the analyzer",
    )
    .native("("),
    dir(
        "php.psalm-suppress",
        LS,
        SUP,
        "psalm",
        Pat::Has("@psalm-suppress"),
        "a `@psalm-suppress` annotation silences the analyzer",
    ),
    dir(
        "php.phpcs-ignore",
        LS,
        SUP,
        "phpcs",
        Pat::WordAny(&["phpcs:ignore", "phpcs:disable", "phpcs:ignoreFile"]),
        "a `phpcs:ignore` comment silences the style checker",
    )
    .native("--"),
    dir(
        "php.coding-standards-ignore",
        LS,
        SUP,
        "phpcs",
        Pat::Has("@codingStandardsIgnore"),
        "a `@codingStandardsIgnore` annotation silences the style checker",
    ),
    code(
        "php.skip",
        SKIP,
        SUP,
        Pat::WordAny(&["markTestSkipped(", "markTestIncomplete("]),
        "`markTestSkipped`/`markTestIncomplete` switches the test off from inside",
    )
    .tool("phpunit"),
    code(
        "php.pest-skip",
        SKIP,
        SUP,
        Pat::WordAny(&["->skip(", "->todo("]),
        "Pest `->skip()`/`->todo()` keeps the test from running",
    )
    .tool("pest")
    .requires(&["test", "it"]),
    check(
        "php.at-suppress",
        ESC,
        EVA,
        php_error_suppression,
        "the `@` operator silences every error the expression raises",
    ),
    code(
        "php.error-reporting-0",
        ESC,
        EVA,
        Pat::Has("error_reporting ( 0 )"),
        "`error_reporting(0)` silences every error",
    ),
    code(
        "php.strict-types-0",
        TYPE,
        EVA,
        Pat::Has("strict_types = 0"),
        "`strict_types=0` turns scalar type checks back into coercions",
    ),
    code(
        "php.throw-stub",
        STUB,
        EVA,
        Pat::Lit("throw new", STUB_PHRASES),
        "a `throw new ...('not implemented')` is a hole presented as an implementation",
    ),
    check(
        "php.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
    dir(
        "php.commented-test",
        SKIP,
        EVA,
        "phpunit",
        Pat::Has("public function test"),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const SWIFT: &[Rule] = &[
    dir(
        "swift.swiftlint-disable",
        LS,
        SUP,
        "swiftlint",
        Pat::Has("swiftlint:disable"),
        "a `swiftlint:disable` comment silences the linter",
    ),
    code(
        "swift.xctskip",
        SKIP,
        SUP,
        Pat::Word("XCTSkip("),
        "`XCTSkip` switches the test off from inside",
    )
    .tool("xctest"),
    code(
        "swift.xctskip-if",
        SKIP,
        EVA,
        Pat::WordAny(&["XCTSkipIf(", "XCTSkipUnless("]),
        "a conditional `XCTSkip` can switch the test off where the gate runs",
    )
    .tool("xctest"),
    code(
        "swift.testing-disabled",
        SKIP,
        SUP,
        Pat::Seq(&["@Test", ".disabled("]),
        "`.disabled()` keeps the test from running",
    )
    .tool("swift-testing"),
    code(
        "swift.known-issue",
        WEAK,
        SUP,
        Pat::Word("withKnownIssue"),
        "`withKnownIssue` turns a failure into an expected one",
    )
    .tool("swift-testing"),
    code(
        "swift.fatal-stub",
        STUB,
        EVA,
        Pat::Lit("fatalError", STUB_PHRASES),
        "a `fatalError(\"not implemented\")` is a hole presented as an implementation",
    ),
    dir(
        "swift.swiftformat-disable",
        LS,
        NOTE,
        "swiftformat",
        Pat::Has("swiftformat:disable"),
        "formatting is switched off; recorded, not a defect by itself",
    ),
    check(
        "swift.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
    dir(
        "swift.commented-test",
        SKIP,
        EVA,
        "xctest",
        Pat::Has("func test"),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const DART: &[Rule] = &[
    dir(
        "dart.ignore",
        LS,
        SUP,
        "dart-analyzer",
        Pat::WordAny(&["ignore:", "ignore_for_file:"]),
        "an `// ignore:` comment silences the analyzer",
    ),
    code(
        "dart.skip",
        SKIP,
        SUP,
        Pat::Word("skip:"),
        "a `skip:` argument keeps the test from running",
    )
    .tool("package:test")
    .except(&["skip: false"]),
    code(
        "dart.unimplemented",
        STUB,
        EVA,
        Pat::Word("UnimplementedError"),
        "`throw UnimplementedError()` is a hole presented as an implementation",
    ),
    dir(
        "dart.coverage-ignore",
        WEAK,
        NOTE,
        "package:coverage",
        Pat::HasAny(&[
            "coverage:ignore-line",
            "coverage:ignore-start",
            "coverage:ignore-file",
        ]),
        "the code is excluded from coverage measurement",
    )
    .near(TEST_MARKERS, EVA),
    dir(
        "dart.commented-test",
        SKIP,
        EVA,
        "package:test",
        Pat::WordAny(&["test(", "testWidgets("]),
        "a commented-out test is presented as coverage it no longer provides",
    ),
];

const LUA: &[Rule] = &[
    dir(
        "lua.luacheck-ignore",
        LS,
        SUP,
        "luacheck",
        Pat::WordAny(&[
            "luacheck: ignore",
            "luacheck: push ignore",
            "luacheck: no",
            "luacheck: push no",
        ]),
        "a `luacheck: ignore` comment silences the linter",
    ),
    dir(
        "lua.selene-allow",
        LS,
        SUP,
        "selene",
        Pat::Has("selene: allow"),
        "a `selene: allow` comment silences the linter",
    ),
    dir(
        "lua.lls-diagnostic",
        LS,
        SUP,
        "lua-language-server",
        Pat::Has("@diagnostic disable"),
        "a `@diagnostic disable` annotation silences the analyzer",
    ),
    code(
        "lua.error-stub",
        STUB,
        EVA,
        Pat::Lit("error", STUB_PHRASES),
        "an `error(\"not implemented\")` is a hole presented as an implementation",
    ),
    code(
        "lua.pending",
        SKIP,
        SUP,
        Pat::Lead(&["pending("]),
        "busted `pending()` registers the test without running it",
    )
    .tool("busted"),
];

const PERL: &[Rule] = &[
    dir(
        "pl.no-critic",
        LS,
        SUP,
        "perlcritic",
        Pat::Has("# no critic"),
        "a `## no critic` comment silences Perl::Critic",
    ),
    code(
        "pl.no-strict",
        ESC,
        SUP,
        Pat::Word("no strict"),
        "`no strict` switches off the checks that catch typos and symbolic references",
    )
    .tool("perl"),
    code(
        "pl.no-warnings",
        LS,
        SUP,
        Pat::Word("no warnings"),
        "`no warnings` silences the interpreter's warnings",
    )
    .tool("perl"),
    code(
        "pl.sig-warn",
        ESC,
        EVA,
        Pat::Has("$SIG{__WARN__} = sub { }"),
        "an empty `__WARN__` handler discards every warning",
    ),
    code(
        "pl.skip-all",
        SKIP,
        SUP,
        Pat::Word("skip_all"),
        "`plan skip_all` switches the whole test file off",
    )
    .tool("test-more"),
    code(
        "pl.skip",
        SKIP,
        SUP,
        Pat::Lead(&["skip"]),
        "`skip` switches tests off",
    )
    .tool("test-more"),
    code(
        "pl.todo",
        WEAK,
        SUP,
        Pat::Has("local $TODO"),
        "a `$TODO` block turns failures into expected ones",
    )
    .tool("test-more"),
    code(
        "pl.yada",
        STUB,
        EVA,
        Pat::Lead(&["..."]),
        "the `...` statement dies with 'Unimplemented'",
    )
    .tool("perl"),
    code(
        "pl.die-stub",
        STUB,
        EVA,
        Pat::Lit("die", STUB_PHRASES),
        "a `die \"not implemented\"` is a hole presented as an implementation",
    ),
];

const R_LANG: &[Rule] = &[
    dir(
        "r.nolint",
        LS,
        SUP,
        "lintr",
        Pat::Word("nolint"),
        "a `# nolint` comment silences the linter",
    ),
    code(
        "r.skip",
        SKIP,
        SUP,
        Pat::Word("skip("),
        "testthat `skip()` switches the test off",
    )
    .tool("testthat"),
    code(
        "r.skip-on-ci",
        SKIP,
        SUP,
        Pat::WordAny(&["skip_on_ci(", "skip_on_covr("]),
        "`skip_on_ci()` switches the test off exactly where the gate runs",
    )
    .tool("testthat"),
    code(
        "r.skip-if",
        SKIP,
        EVA,
        Pat::WordAny(&["skip_if(", "skip_if_not("]),
        "a conditional skip can switch the test off where the gate runs",
    )
    .tool("testthat"),
    code(
        "r.suppress",
        ESC,
        EVA,
        Pat::WordAny(&["suppressWarnings(", "suppressMessages("]),
        "`suppressWarnings` hides the warnings the call raises",
    ),
    code(
        "r.try-silent",
        ESC,
        EVA,
        Pat::Has("silent = TRUE"),
        "`try(..., silent = TRUE)` swallows the error",
    ),
    code(
        "r.stop-stub",
        STUB,
        EVA,
        Pat::Lit("stop", STUB_PHRASES),
        "a `stop(\"not implemented\")` is a hole presented as an implementation",
    ),
    code(
        "r.not-yet-implemented",
        STUB,
        EVA,
        Pat::Word(".NotYetImplemented("),
        "`.NotYetImplemented()` is a hole presented as an implementation",
    ),
];

const JULIA: &[Rule] = &[
    code(
        "jl.test-skip",
        SKIP,
        SUP,
        Pat::Word("@test_skip"),
        "`@test_skip` records the test as skipped",
    )
    .tool("Test"),
    code(
        "jl.test-broken",
        WEAK,
        SUP,
        Pat::Word("@test_broken"),
        "`@test_broken` turns a failure into an expected one",
    )
    .tool("Test"),
    code(
        "jl.test-true",
        WEAK,
        SUP,
        Pat::Word("@test true"),
        "an assertion on a constant cannot fail",
    )
    .tool("Test"),
    code(
        "jl.error-stub",
        STUB,
        EVA,
        Pat::Lit("error", STUB_PHRASES),
        "an `error(\"not implemented\")` is a hole presented as an implementation",
    ),
    dir(
        "jl.format-off",
        LS,
        NOTE,
        "JuliaFormatter",
        Pat::Has("format: off"),
        "formatting is switched off; recorded, not a defect by itself",
    ),
];

const HASKELL: &[Rule] = &[
    check(
        "hs.ghc-options",
        LS,
        SUP,
        hs_ghc_options_suppress,
        "`OPTIONS_GHC -w`/`-Wno-*` silences compiler warnings for the module",
    )
    .tool("ghc"),
    dir(
        "hs.hlint-ignore",
        LS,
        SUP,
        "hlint",
        Pat::HasAny(&["hlint ignore", "# hlint ignore"]),
        "an `HLINT ignore` pragma silences the linter",
    )
    .ci(),
    rule(
        "hs.ann-hlint",
        LS,
        SUP,
        Site::Comment,
        Pat::Seq(&["ANN", "HLint: ignore"]),
        "an `ANN \"HLint: ignore\"` annotation silences the linter",
    )
    .tool("hlint"),
    code(
        "hs.undefined",
        STUB,
        EVA,
        Pat::Word("undefined"),
        "`undefined` is a hole presented as an implementation",
    ),
    code(
        "hs.error-stub",
        STUB,
        EVA,
        Pat::Lit("error", STUB_PHRASES),
        "an `error \"TODO\"` is a hole presented as an implementation",
    ),
    code(
        "hs.hspec-pending",
        SKIP,
        SUP,
        Pat::WordAny(&["pending", "pendingWith", "xit", "xdescribe", "xspecify"]),
        "a pending or `x`-prefixed spec is not run",
    )
    .tool("hspec"),
    code(
        "hs.unsafe",
        ESC,
        EVA,
        Pat::WordAny(&["unsafePerformIO", "unsafeCoerce"]),
        "`unsafePerformIO`/`unsafeCoerce` escapes the type system's guarantees",
    ),
    code(
        "hs.ignore-test",
        SKIP,
        SUP,
        Pat::Word("ignoreTest"),
        "`ignoreTest` keeps the test from running",
    )
    .tool("tasty"),
];

const OCAML: &[Rule] = &[
    code(
        "ml.warning-attr",
        LS,
        SUP,
        Pat::Lit("warning", &["-"]),
        "a `[@warning \"-N\"]` attribute disables a compiler warning",
    )
    .tool("ocaml")
    .requires(&["[@"]),
    code(
        "ml.failwith-stub",
        STUB,
        EVA,
        Pat::Lit("failwith", STUB_PHRASES),
        "a `failwith \"todo\"` is a hole presented as an implementation",
    ),
    code(
        "ml.obj-magic",
        TYPE,
        EVA,
        Pat::Word("Obj.magic"),
        "`Obj.magic` defeats the type system",
    ),
    code(
        "ml.alcotest-skip",
        SKIP,
        SUP,
        Pat::Word("Alcotest.skip"),
        "`Alcotest.skip` switches the test off",
    )
    .tool("alcotest"),
];

const ELIXIR: &[Rule] = &[
    dir(
        "ex.credo-disable",
        LS,
        SUP,
        "credo",
        Pat::Has("credo:disable"),
        "a `credo:disable` comment silences the linter",
    ),
    code(
        "ex.tag-skip",
        SKIP,
        SUP,
        Pat::WordAny(&[
            "@tag :skip",
            "@moduletag :skip",
            "@describetag :skip",
            "@tag skip:",
            "@tag :pending",
        ]),
        "a `:skip` tag keeps the test from running",
    )
    .tool("exunit"),
    code(
        "ex.dialyzer",
        TYPE,
        SUP,
        Pat::Word("@dialyzer"),
        "a `@dialyzer` attribute silences the type analyzer",
    )
    .tool("dialyxir"),
    code(
        "ex.raise-stub",
        STUB,
        EVA,
        Pat::Lit("raise", STUB_PHRASES),
        "a `raise \"not implemented\"` is a hole presented as an implementation",
    ),
    dir(
        "ex.coveralls-ignore",
        WEAK,
        NOTE,
        "excoveralls",
        Pat::Has("coveralls-ignore"),
        "the code is excluded from coverage measurement",
    )
    .near(TEST_MARKERS, EVA),
    code(
        "ex.credo-config-off",
        CFG,
        SUP,
        Pat::Seq(&["Credo.Check.", ", false"]),
        "a check set to `false` in `.credo.exs` is switched off",
    )
    .tool("credo"),
];

const ERLANG: &[Rule] = &[
    code(
        "erl.nowarn",
        LS,
        SUP,
        Pat::Seq(&["-compile", "nowarn_"]),
        "`-compile(nowarn_...)` silences a compiler warning",
    )
    .tool("erlc"),
    code(
        "erl.dialyzer",
        TYPE,
        SUP,
        Pat::Has("-dialyzer"),
        "a `-dialyzer` attribute silences the type analyzer",
    )
    .tool("dialyzer"),
    code(
        "erl.elvis",
        LS,
        SUP,
        Pat::Has("-elvis("),
        "an `-elvis` attribute disables style rules",
    )
    .tool("elvis"),
    code(
        "erl.not-implemented",
        STUB,
        EVA,
        Pat::WordAny(&["not_implemented", "notimplemented"]),
        "a `not_implemented` error is a hole presented as an implementation",
    ),
    code(
        "erl.ct-skip",
        SKIP,
        SUP,
        Pat::Has("{ skip ,"),
        "returning `{skip, Reason}` switches the test off",
    )
    .tool("common_test"),
    code(
        "erl.assert-true",
        WEAK,
        SUP,
        Pat::Has("?assert(true)"),
        "an assertion on a constant cannot fail",
    )
    .tool("eunit"),
];

const CLOJURE: &[Rule] = &[
    code(
        "clj.discard-test",
        SKIP,
        SUP,
        Pat::HasAny(&["#_ (deftest", "#_ (testing"]),
        "`#_` discards the test form: it is never compiled",
    )
    .tool("clojure.test"),
    code(
        "clj.discard-assert",
        WEAK,
        SUP,
        Pat::Has("#_ (is"),
        "`#_` discards the assertion form",
    )
    .tool("clojure.test"),
    code(
        "clj.skip-meta",
        SKIP,
        SUP,
        Pat::WordAny(&["^:skip", "^:kaocha/skip", "^:kaocha/pending"]),
        "skip metadata keeps the test from running",
    )
    .tool("kaocha"),
    code(
        "clj.comment-test",
        SKIP,
        SUP,
        Pat::Seq(&["(comment", "(deftest"]),
        "a test inside `(comment ...)` is never compiled",
    )
    .tool("clojure.test"),
    code(
        "clj.is-true",
        WEAK,
        SUP,
        Pat::Has("(is true)"),
        "an assertion on a constant cannot fail",
    )
    .tool("clojure.test"),
    code(
        "clj.kondo-ignore",
        LS,
        SUP,
        Pat::WordAny(&["#_:clj-kondo/ignore", "#_{:clj-kondo/ignore"]),
        "a `clj-kondo/ignore` annotation silences the linter",
    )
    .tool("clj-kondo"),
    code(
        "clj.throw-stub",
        STUB,
        EVA,
        Pat::Lit("throw", STUB_PHRASES),
        "a `(throw (ex-info \"not implemented\" ...))` is a hole presented as an implementation",
    ),
];

const ZIG: &[Rule] = &[
    check("zig.discard-call", ESC, EVA, zig_discard_call, "`_ = f();` discards a result, and with it the error it may carry"),
    code("zig.catch-unreachable", ESC, EVA, Pat::Has("catch unreachable"), "`catch unreachable` asserts an error cannot happen; in release builds it is undefined behavior"),
    code("zig.catch-empty", ESC, EVA, Pat::Has("catch { }"), "`catch {}` swallows the error"),
    code("zig.panic-stub", STUB, EVA, Pat::Lit("@panic", STUB_PHRASES), "a `@panic(\"TODO\")` is a hole presented as an implementation"),
    check("zig.unreachable-body", STUB, EVA, zig_unreachable_body, "a function whose body is `unreachable;` is a hole presented as an implementation"),
    code("zig.skip-test", SKIP, SUP, Pat::Has("error.SkipZigTest"), "returning `error.SkipZigTest` switches the test off").tool("zig test"),
    code("zig.runtime-safety-off", ESC, EVA, Pat::Has("@setRuntimeSafety ( false )"), "`@setRuntimeSafety(false)` removes bounds and overflow checks"),
];

const NIM: &[Rule] = &[
    code(
        "nim.warning-off",
        LS,
        SUP,
        Pat::Seq(&["{.", "warning", ": off"]),
        "a `{.warning[X]: off.}` pragma silences compiler warnings",
    )
    .tool("nim"),
    code(
        "nim.hint-off",
        LS,
        SUP,
        Pat::Seq(&["{.", "hint", ": off"]),
        "a `{.hint[X]: off.}` pragma silences compiler hints",
    )
    .tool("nim"),
    code(
        "nim.checks-off",
        ESC,
        EVA,
        Pat::Seq(&["{.", "checks : off"]),
        "switching runtime checks off removes bounds and overflow checks",
    )
    .tool("nim")
    .ci(),
    check(
        "nim.discard-body",
        STUB,
        EVA,
        nim_discard_body,
        "a routine whose body is `discard` is an empty function presented as an implementation",
    ),
    code(
        "nim.raise-stub",
        STUB,
        EVA,
        Pat::Lit("raise", STUB_PHRASES),
        "a `raise newException(..., \"not implemented\")` is a hole presented as an implementation",
    ),
    code(
        "nim.skip",
        SKIP,
        SUP,
        Pat::Word("skip("),
        "unittest `skip()` switches the test off",
    )
    .tool("unittest"),
];

const SOLIDITY: &[Rule] = &[
    dir(
        "sol.solhint-disable",
        LS,
        SUP,
        "solhint",
        Pat::Has("solhint-disable"),
        "a `solhint-disable` comment silences the linter",
    ),
    dir(
        "sol.slither-disable",
        LS,
        SUP,
        "slither",
        Pat::Has("slither-disable"),
        "a `slither-disable` comment silences the security analyzer",
    ),
    dir(
        "sol.forge-lint",
        LS,
        SUP,
        "forge",
        Pat::Has("forge-lint: disable"),
        "a `forge-lint: disable` comment silences the linter",
    ),
    dir(
        "sol.aderyn-ignore",
        LS,
        SUP,
        "aderyn",
        Pat::Has("aderyn-ignore"),
        "an `aderyn-ignore` comment silences the security analyzer",
    ),
    code(
        "sol.unchecked",
        ESC,
        EVA,
        Pat::Word("unchecked {"),
        "an `unchecked` block removes overflow checks from arithmetic that moves value",
    )
    .tool("solc"),
    code(
        "sol.vm-skip",
        SKIP,
        SUP,
        Pat::Has("vm.skip ( true )"),
        "`vm.skip(true)` switches the test off",
    )
    .tool("foundry"),
    code(
        "sol.revert-stub",
        STUB,
        EVA,
        Pat::Lit("revert", STUB_PHRASES),
        "a `revert(\"not implemented\")` is a hole presented as an implementation",
    ),
];

const SHELL: &[Rule] = &[
    dir(
        "sh.shellcheck-disable",
        LS,
        SUP,
        "shellcheck",
        Pat::Has("shellcheck disable"),
        "a `# shellcheck disable=` comment silences the linter",
    ),
    code(
        "sh.shellcheckrc-disable",
        LS,
        SUP,
        Pat::Lead(&["disable = SC"]),
        "a `disable=` line in `.shellcheckrc` silences the check everywhere",
    )
    .tool("shellcheck"),
    code(
        "sh.set-plus-e",
        ESC,
        EVA,
        Pat::WordAny(&["set +e", "set +o errexit"]),
        "`set +e` lets every later command fail without stopping the script",
    )
    .tool("bash"),
    text(
        "sh.or-true",
        WEAK,
        FRAUD,
        Pat::WordAny(&["|| true", "|| :", "|| exit 0"]),
        "`|| true` after a test or lint command turns its failure into success",
    )
    .requires(GATE_CMDS)
    .overlap("cheat::scan (gate scripts)"),
    text(
        "sh.devnull",
        WEAK,
        EVA,
        Pat::HasAny(&["2>/dev/null", "&>/dev/null", ">/dev/null 2>&1"]),
        "discarding a test command's output hides why it failed",
    )
    .requires(GATE_CMDS),
    text(
        "sh.cap-lints",
        CFG,
        FRAUD,
        Pat::Has("--cap-lints"),
        "`--cap-lints` caps the severity of every lint the compiler reports",
    )
    .tool("rustc")
    .overlap("cheat::scan (gate-side --cap-lints)"),
    text(
        "sh.rust-allow-flag",
        LS,
        SUP,
        Pat::Flag("-A"),
        "an `-A` flag allows a lint for the whole build",
    )
    .tool("rustc")
    .requires(RUST_TOOLS),
    text(
        "sh.rust-allow-long",
        LS,
        SUP,
        Pat::Flag("--allow"),
        "an `--allow` flag allows a lint for the whole build",
    )
    .tool("rustc")
    .requires(RUST_TOOLS),
    text(
        "sh.rust-allow-warnings",
        LS,
        SUP,
        Pat::Has("-Awarnings"),
        "`-Awarnings` allows every warning",
    )
    .tool("rustc"),
    text(
        "sh.no-verify",
        CFG,
        FRAUD,
        Pat::Flag("--no-verify"),
        "`--no-verify` skips the pre-commit and pre-push hooks",
    )
    .tool("git")
    .requires(&["git"]),
    text(
        "sh.precommit-skip-env",
        CFG,
        SUP,
        Pat::Has("SKIP="),
        "`SKIP=` names pre-commit hooks not to run",
    )
    .tool("pre-commit")
    .requires(&["git", "pre-commit"]),
    text(
        "sh.gradle-exclude-test",
        SKIP,
        FRAUD,
        Pat::Has("-x test"),
        "`-x test` excludes the test task from the build",
    )
    .tool("gradle")
    .requires(&["gradle", "gradlew"]),
    text(
        "sh.maven-skip",
        SKIP,
        FRAUD,
        Pat::WordAny(&[
            "-DskipTests",
            "-Dmaven.test.skip=true",
            "-Dmaven.test.failure.ignore=true",
        ]),
        "the Maven property skips the tests or ignores their failure",
    )
    .tool("maven"),
    text(
        "sh.cargo-skip",
        SKIP,
        EVA,
        Pat::Flag("--skip"),
        "`--skip` filters tests out of the run",
    )
    .tool("cargo")
    .requires(&["cargo"]),
    text(
        "sh.pytest-deselect",
        SKIP,
        SUP,
        Pat::WordAny(&["--deselect", "--ignore=", "--ignore-glob"]),
        "`--deselect`/`--ignore` removes tests from collection",
    )
    .tool("pytest")
    .requires(&["pytest", "py.test"]),
    text(
        "sh.pytest-k-not",
        SKIP,
        EVA,
        Pat::SeqWord(&["-k", "not"]),
        "`-k \"not ...\"` deselects tests by name",
    )
    .tool("pytest")
    .requires(&["pytest", "py.test"]),
    text(
        "sh.make-ignore",
        WEAK,
        FRAUD,
        Pat::Flag("-i"),
        "`make -i` ignores every recipe failure",
    )
    .tool("make")
    .requires(&["make", "MAKE"]),
    text(
        "sh.make-ignore-long",
        WEAK,
        FRAUD,
        Pat::Has("--ignore-errors"),
        "`--ignore-errors` ignores every recipe failure",
    )
    .tool("make"),
    code(
        "sh.bats-skip",
        SKIP,
        SUP,
        Pat::Lead(&["skip"]),
        "bats `skip` switches the test off from inside",
    )
    .tool("bats")
    .except(&["skip:"]),
];

/// Test-runner and linter flags that weaken the gate wherever a command line
/// appears: shell, CI, Makefiles, Dockerfiles, `package.json`, `addopts`.
const GATE_FLAGS: &[Rule] = &[
    text(
        "flag.pass-with-no-tests",
        WEAK,
        SUP,
        Pat::WordAny(&["--passWithNoTests", "--pass-with-no-tests"]),
        "`--passWithNoTests` makes a run that found no tests a pass",
    )
    .tool("jest"),
    text(
        "flag.pytest-no-warnings",
        LS,
        SUP,
        Pat::HasAny(&["--disable-warnings", "-p no:warnings"]),
        "disabling pytest's warning capture hides every warning the suite raises",
    )
    .tool("pytest"),
    text(
        "flag.python-w-ignore",
        LS,
        SUP,
        Pat::HasAny(&["-W ignore", "-Wignore", "PYTHONWARNINGS=ignore"]),
        "`-W ignore` silences every Python warning",
    )
    .tool("python"),
];

/// Compiler flags that silence or demote diagnostics.
const COMPILER_FLAGS: &[Rule] = &[
    text(
        "cc.wno-error",
        CFG,
        SUP,
        Pat::Has("-Wno-error"),
        "`-Wno-error` demotes errors back to warnings",
    )
    .tool("gcc"),
    text(
        "cc.wno",
        LS,
        SUP,
        Pat::Has("-Wno-"),
        "a `-Wno-*` flag disables a compiler warning",
    )
    .tool("gcc")
    .except(&["-Wno-error"]),
    text(
        "cc.w",
        LS,
        SUP,
        Pat::Flag("-w"),
        "`-w` disables every compiler warning",
    )
    .tool("gcc")
    .requires(COMPILERS),
    text(
        "cc.fpermissive",
        TYPE,
        EVA,
        Pat::Has("-fpermissive"),
        "`-fpermissive` demotes type errors to warnings",
    )
    .tool("g++"),
    text(
        "cc.msvc-w0",
        LS,
        SUP,
        Pat::Flag("/W0"),
        "`/W0` disables every compiler warning",
    )
    .tool("msvc"),
];

const POWERSHELL: &[Rule] = &[
    code(
        "ps.suppress-message",
        LS,
        SUP,
        Pat::HasAny(&["CodeAnalysis.SuppressMessage", "[SuppressMessage"]),
        "a `SuppressMessageAttribute` silences PSScriptAnalyzer",
    )
    .tool("psscriptanalyzer")
    .ci()
    .native("Justification"),
    text(
        "ps.erroraction",
        ESC,
        EVA,
        Pat::HasAny(&[
            "-ErrorAction SilentlyContinue",
            "-ErrorAction Ignore",
            "-EA SilentlyContinue",
            "-EA 0",
            "ErrorActionPreference = 'SilentlyContinue'",
            "ErrorActionPreference = \"SilentlyContinue\"",
        ]),
        "`SilentlyContinue` discards the error the command raises",
    )
    .ci(),
    code(
        "ps.strictmode-off",
        ESC,
        SUP,
        Pat::Has("Set-StrictMode -Off"),
        "`Set-StrictMode -Off` switches off uninitialized-variable and property checks",
    )
    .ci(),
    code(
        "ps.pester-skip",
        SKIP,
        SUP,
        Pat::WordAny(&["-Skip", "-Pending"]),
        "Pester `-Skip`/`-Pending` keeps the test from running",
    )
    .tool("pester")
    .ci()
    .requires(&["It", "Describe", "Context"]),
    code(
        "ps.set-itresult",
        SKIP,
        SUP,
        Pat::Has("Set-ItResult"),
        "`Set-ItResult` marks the test skipped, pending or inconclusive",
    )
    .tool("pester")
    .ci(),
    code(
        "ps.throw-stub",
        STUB,
        EVA,
        Pat::Lit("throw", STUB_PHRASES),
        "a `throw 'not implemented'` is a hole presented as an implementation",
    )
    .ci(),
    code(
        "ps.not-implemented",
        STUB,
        EVA,
        Pat::Word("NotImplementedException"),
        "a `NotImplementedException` is a hole presented as an implementation",
    ),
    code(
        "ps.exclude-rules",
        CFG,
        SUP,
        Pat::Word("ExcludeRules"),
        "`ExcludeRules` in analyzer settings switches rules off",
    )
    .tool("psscriptanalyzer")
    .ci(),
    check(
        "ps.empty-catch",
        ESC,
        EVA,
        empty_catch,
        "an empty `catch` swallows every error it was written to handle",
    ),
];

const SQL: &[Rule] = &[
    dir(
        "sql.noqa",
        LS,
        SUP,
        "sqlfluff",
        Pat::Word("noqa"),
        "a `-- noqa` comment silences the linter",
    )
    .ci(),
    code(
        "sql.exclude-rules",
        CFG,
        SUP,
        Pat::Lead(&["exclude_rules"]),
        "`exclude_rules` in `.sqlfluff` switches rules off",
    )
    .tool("sqlfluff"),
    code(
        "sql.pgtap-skip",
        SKIP,
        SUP,
        Pat::Word("skip("),
        "pgTAP `skip()` switches tests off",
    )
    .tool("pgtap"),
    code(
        "sql.pgtap-todo",
        WEAK,
        SUP,
        Pat::WordAny(&["todo(", "todo_start("]),
        "pgTAP `todo` turns failures into expected ones",
    )
    .tool("pgtap"),
    text(
        "sql.sql-mode",
        ESC,
        EVA,
        Pat::Has("sql_mode = ''"),
        "an empty `sql_mode` switches MySQL's strict checks off",
    )
    .ci(),
    code(
        "sql.fk-checks",
        ESC,
        EVA,
        Pat::Has("FOREIGN_KEY_CHECKS = 0"),
        "`FOREIGN_KEY_CHECKS = 0` switches referential integrity off",
    )
    .ci(),
    code(
        "sql.raise-stub",
        STUB,
        EVA,
        Pat::Lit("raise", STUB_PHRASES),
        "a `RAISE EXCEPTION 'not implemented'` is a hole presented as an implementation",
    )
    .ci(),
];

/// Infrastructure-as-code scanners, which read comments in HCL, YAML and
/// Dockerfiles alike.
const IAC: &[Rule] = &[
    dir(
        "iac.tfsec-ignore",
        LS,
        SUP,
        "tfsec",
        Pat::Has("tfsec:ignore"),
        "a `tfsec:ignore` comment silences the security scanner",
    ),
    dir(
        "iac.checkov-skip",
        LS,
        SUP,
        "checkov",
        Pat::Has("checkov:skip"),
        "a `checkov:skip` comment silences the security scanner",
    )
    .native(":"),
    dir(
        "iac.trivy-ignore",
        LS,
        SUP,
        "trivy",
        Pat::Has("trivy:ignore"),
        "a `trivy:ignore` comment silences the security scanner",
    ),
    dir(
        "iac.kics-ignore",
        LS,
        SUP,
        "kics",
        Pat::HasAny(&["kics-scan ignore", "kics-scan disable"]),
        "a `kics-scan` comment silences the security scanner",
    ),
    dir(
        "iac.terrascan-skip",
        LS,
        SUP,
        "terrascan",
        Pat::Has("ts:skip"),
        "a `ts:skip` comment silences the security scanner",
    ),
];

const TERRAFORM: &[Rule] = &[dir(
    "tf.tflint-ignore",
    LS,
    SUP,
    "tflint",
    Pat::Has("tflint-ignore"),
    "a `tflint-ignore` comment silences the linter",
)];

const DOCKERFILE: &[Rule] = &[
    dir(
        "docker.hadolint-ignore",
        LS,
        SUP,
        "hadolint",
        Pat::HasAny(&["hadolint ignore", "hadolint global ignore"]),
        "a `# hadolint ignore=` comment silences the linter",
    ),
    code(
        "docker.hadolint-config",
        CFG,
        SUP,
        Pat::Lead(&["ignored:"]),
        "an `ignored:` list in `.hadolint.yaml` switches rules off",
    )
    .tool("hadolint"),
];

const MAKEFILE: &[Rule] = &[
    check(
        "make.dash-prefix",
        WEAK,
        FRAUD,
        make_dash_gate,
        "a `-` recipe prefix ignores the failure of the test or lint command it runs",
    )
    .tool("make"),
    code(
        "make.ignore-target",
        WEAK,
        FRAUD,
        Pat::Lead(&[".IGNORE"]),
        "`.IGNORE:` ignores every recipe failure",
    )
    .tool("make"),
    text(
        "make.makeflags-i",
        WEAK,
        FRAUD,
        Pat::Flag("-i"),
        "`-i` in `MAKEFLAGS` ignores every recipe failure",
    )
    .tool("make")
    .requires(&["MAKEFLAGS"]),
    check(
        "make.noop-gate",
        WEAK,
        FRAUD,
        make_noop_gate_recipe,
        "a test/lint/gate target whose recipe does nothing passes without checking anything",
    )
    .tool("make"),
];

const YAML: &[Rule] = &[
    dir(
        "yaml.yamllint-disable",
        LS,
        SUP,
        "yamllint",
        Pat::Has("yamllint disable"),
        "a `yamllint disable` comment silences the linter",
    ),
    dir(
        "yaml.ansible-noqa",
        LS,
        SUP,
        "ansible-lint",
        Pat::Word("noqa"),
        "a `# noqa` comment silences ansible-lint",
    ),
    text(
        "yaml.kube-linter",
        LS,
        SUP,
        Pat::Has("ignore-check.kube-linter.io"),
        "an `ignore-check.kube-linter.io` annotation silences the linter",
    )
    .tool("kube-linter"),
];

const CI: &[Rule] = &[
    text(
        "ci.continue-on-error",
        WEAK,
        FRAUD,
        Pat::HasAny(&["continue-on-error: true", "continueOnError: true"]),
        "`continue-on-error: true` lets the job fail without failing the pipeline",
    )
    .tool("github-actions"),
    text(
        "ci.allow-failure",
        WEAK,
        FRAUD,
        Pat::Has("allow_failure: true"),
        "`allow_failure: true` lets the job fail without failing the pipeline",
    )
    .tool("gitlab-ci"),
    code(
        "ci.allow-failures",
        WEAK,
        SUP,
        Pat::Lead(&["allow_failures:"]),
        "`allow_failures` lets matrix jobs fail without failing the build",
    )
    .tool("travis"),
    text(
        "ci.if-false",
        SKIP,
        FRAUD,
        Pat::HasAny(&["if: false", "if: ${{ false }}", "if: 'false'"]),
        "`if: false` means the job or step never runs",
    )
    .tool("github-actions"),
    text(
        "ci.jenkins-catch",
        WEAK,
        FRAUD,
        Pat::Seq(&["catchError", "SUCCESS"]),
        "`catchError(buildResult: 'SUCCESS')` reports a failed stage as passing",
    )
    .tool("jenkins"),
    text(
        "ci.jenkins-status",
        WEAK,
        EVA,
        Pat::Has("returnStatus: true"),
        "`returnStatus: true` turns a failing command into a number nobody checks",
    )
    .tool("jenkins"),
    text(
        "ci.precommit-manual",
        SKIP,
        EVA,
        Pat::Has("stages: [manual]"),
        "a hook staged `manual` never runs on commit",
    )
    .tool("pre-commit"),
];

const CARGO: &[Rule] = &[
    code("cargo.autotests-off", SKIP, SUP, Pat::Has("autotests = false"), "`autotests = false` stops `tests/*.rs` from being compiled").tool("cargo").overlap("cheat::scan_manifest"),
    code("cargo.test-false", SKIP, SUP, Pat::Lead(&["test = false"]), "`test = false` stops `cargo test` running the target's tests").tool("cargo").overlap("cheat::scan_manifest"),
    code("cargo.doctest-false", SKIP, SUP, Pat::Lead(&["doctest = false"]), "`doctest = false` stops documentation examples from being tested").tool("cargo").waiver(Waiver::Reason),
    check("cargo.test-harness-false", SKIP, SUP, cargo_test_harness_false, "`harness = false` on a test target runs its `main` instead of its `#[test]` functions").tool("cargo").waiver(Waiver::Reason),
    check("cargo.test-required-features", SKIP, EVA, cargo_test_required_features, "`required-features` on a test target silently skips it when the feature is off").tool("cargo").waiver(Waiver::Reason),
    code("cargo.cap-lints", CFG, FRAUD, Pat::Has("cap-lints"), "`cap-lints` caps the severity of every lint").tool("rustc").overlap("cheat::scan_manifest"),
    check("cargo.lints-lowered", CFG, FRAUD, cargo_lints_lowered, "a `[lints]` entry set to `allow` or `warn` lowers the gate it was written to raise").tool("cargo").overlap("cheat::scan_manifest"),
    text("cargo.rustflags-allow", CFG, FRAUD, Pat::Flag("-A"), "rustflags carrying `-A` allow a lint crate-wide").tool("rustc").requires(&["rustflags", "RUSTFLAGS"]).overlap("cheat::scan_manifest"),
    check("cargo.test-profile-weakened", WEAK, EVA, cargo_test_profile_weakened, "turning debug assertions or overflow checks off in the test profile removes checks the tests rely on").tool("cargo"),
    check("cargo.alias-gate", CFG, FRAUD, cargo_alias_shadows_gate, "an `[alias]` named after a gate command (`clippy`, `test`) replaces what the gate runs").tool("cargo"),
    code("cargo.rustc-wrapper", CFG, EVA, Pat::Lead(&["rustc-wrapper", "rustc-workspace-wrapper"]), "a rustc wrapper sees, and can rewrite, every compiler and clippy invocation").tool("cargo"),
];

const CLIPPY: &[Rule] = &[
    check(
        "clippy.threshold-raised",
        CFG,
        SUP,
        clippy_threshold_raised,
        "a clippy threshold raised to three times its default or more silences the lint it bounds",
    )
    .tool("clippy")
    .overlap("cheat::scan_clippy_toml (values of 1000 and up)"),
    code(
        "clippy.allow-in-tests",
        CFG,
        NOTE,
        Pat::Seq(&["allow-", "-in-tests = true"]),
        "a project-wide test exemption; recorded, not a defect by itself",
    )
    .tool("clippy"),
];

const PYTHON_CONFIG: &[Rule] = &[
    code(
        "pycfg.ignore",
        LS,
        SUP,
        Pat::Lead(&[
            "ignore =",
            "extend-ignore =",
            "per-file-ignores",
            "extend-per-file-ignores",
        ]),
        "an `ignore` list switches linter rules off",
    )
    .tool("ruff"),
    code(
        "pycfg.ignore-errors",
        CFG,
        FRAUD,
        Pat::Has("ignore_errors = true"),
        "`ignore_errors = true` turns the checker's errors off",
    )
    .tool("mypy")
    .ci(),
    code(
        "pycfg.ignore-outcome",
        WEAK,
        FRAUD,
        Pat::Has("ignore_outcome = true"),
        "`ignore_outcome = true` makes a failing tox environment pass",
    )
    .tool("tox")
    .ci(),
    code(
        "pycfg.mypy-lax",
        TYPE,
        EVA,
        Pat::WordAny(&[
            "disallow_untyped_defs = false",
            "check_untyped_defs = false",
            "strict = false",
            "follow_imports = skip",
        ]),
        "the setting lowers how much of the code the type checker reads",
    )
    .tool("mypy")
    .ci(),
    code(
        "pycfg.disable-error-code",
        TYPE,
        SUP,
        Pat::Lead(&["disable_error_code", "disable-error-code"]),
        "`disable_error_code` silences an error class project-wide",
    )
    .tool("mypy"),
    text(
        "pycfg.pyright-off",
        TYPE,
        FRAUD,
        Pat::Seq(&["typeCheckingMode", "off"]),
        "`typeCheckingMode = \"off\"` turns the type checker off",
    )
    .tool("pyright"),
    text(
        "pycfg.pyright-basic",
        TYPE,
        EVA,
        Pat::Seq(&["typeCheckingMode", "basic"]),
        "`typeCheckingMode = \"basic\"` lowers the checking level",
    )
    .tool("pyright"),
    check(
        "pycfg.pyright-report-off",
        TYPE,
        SUP,
        pyright_report_off,
        "a `reportX = false` setting switches a type check off project-wide",
    )
    .tool("pyright"),
    code(
        "pycfg.pylint-disable",
        LS,
        SUP,
        Pat::Lead(&["disable ="]),
        "a pylint `disable` list switches checks off",
    )
    .tool("pylint"),
    check(
        "pycfg.fail-under-zero",
        CFG,
        FRAUD,
        fail_under_zero,
        "a `fail_under` of zero or less is a gate that cannot fail",
    )
    .tool("coverage.py"),
    code(
        "pycfg.coverage-omit",
        WEAK,
        EVA,
        Pat::Lead(&["omit =", "exclude_lines", "exclude_also"]),
        "omitting files or lines from coverage measurement",
    )
    .tool("coverage.py")
    .waiver(Waiver::Reason),
    text(
        "pycfg.filterwarnings-ignore",
        LS,
        SUP,
        Pat::SeqWord(&["filterwarnings", "ignore"]),
        "a `filterwarnings = ignore` entry hides warnings the suite raises",
    )
    .tool("pytest"),
    text(
        "pycfg.filter-ignore-entry",
        LS,
        SUP,
        Pat::Has("ignore::"),
        "an `ignore::` warning filter hides warnings the suite raises",
    )
    .tool("pytest"),
    code(
        "pycfg.xfail-strict-off",
        WEAK,
        EVA,
        Pat::Has("xfail_strict = false"),
        "non-strict xfail lets a fixed test pass silently",
    )
    .tool("pytest")
    .ci(),
    code(
        "pycfg.norecursedirs-tests",
        SKIP,
        SUP,
        Pat::Seq(&["norecursedirs", "test"]),
        "`norecursedirs` naming test directories removes them from collection",
    )
    .tool("pytest"),
    code(
        "pycfg.bandit-skips",
        LS,
        SUP,
        Pat::Lead(&["skips ="]),
        "a bandit `skips` list switches security checks off",
    )
    .tool("bandit"),
];

const JS_LINT_CONFIG: &[Rule] = &[
    check(
        "jscfg.rule-off",
        CFG,
        SUP,
        config_rule_off,
        "a rule set to `\"off\"`/`0` is switched off for the project",
    )
    .tool("eslint"),
    text(
        "jscfg.ignore-all",
        CFG,
        FRAUD,
        Pat::HasAny(&["\"**\"", "\"**/*\"", "'**'", "'**/*'"]),
        "an ignore pattern that matches everything lints nothing",
    )
    .tool("eslint")
    .requires(&["ignorePatterns", "ignores", "ignore"]),
    text(
        "jscfg.unused-directives-off",
        CFG,
        EVA,
        Pat::Seq(&["reportUnusedDisableDirectives", "off"]),
        "unused disable directives are no longer reported, so stale suppressions accumulate",
    )
    .tool("eslint"),
    check(
        "jscfg.biome-linter-off",
        CFG,
        FRAUD,
        biome_linter_disabled,
        "`\"linter\": { \"enabled\": false }` turns the linter off",
    )
    .tool("biome"),
    text(
        "jscfg.recommended-false",
        CFG,
        SUP,
        Pat::Has("\"recommended\" : false"),
        "`\"recommended\": false` drops the default rule set",
    )
    .tool("biome"),
];

const TS_CONFIG: &[Rule] = &[
    text(
        "tscfg.strict-off",
        TYPE,
        SUP,
        Pat::HasAny(&[
            "\"strict\" : false",
            "\"noImplicitAny\" : false",
            "\"strictNullChecks\" : false",
            "\"strictFunctionTypes\" : false",
            "\"strictBindCallApply\" : false",
            "\"strictPropertyInitialization\" : false",
            "\"noImplicitThis\" : false",
            "\"alwaysStrict\" : false",
            "\"useUnknownInCatchVariables\" : false",
        ]),
        "switching a strict-family flag off lowers what the compiler checks",
    )
    .tool("tsc"),
    text(
        "tscfg.suppress",
        TYPE,
        SUP,
        Pat::HasAny(&[
            "\"suppressImplicitAnyIndexErrors\" : true",
            "\"suppressExcessPropertyErrors\" : true",
            "\"noStrictGenericChecks\" : true",
        ]),
        "the option suppresses a class of type errors",
    )
    .tool("tsc"),
    text(
        "tscfg.allow-unreachable",
        LS,
        EVA,
        Pat::HasAny(&[
            "\"allowUnreachableCode\" : true",
            "\"allowUnusedLabels\" : true",
        ]),
        "the option silences dead-code diagnostics",
    )
    .tool("tsc"),
    text(
        "tscfg.skip-lib-check",
        TYPE,
        NOTE,
        Pat::Has("\"skipLibCheck\" : true"),
        "declaration files are not type-checked; common, recorded",
    )
    .tool("tsc"),
];

const NODE_PACKAGE: &[Rule] = &[
    check(
        "npm.noop-script",
        WEAK,
        FRAUD,
        npm_noop_script,
        "a test/lint script that only exits 0 or echoes passes without checking anything",
    )
    .tool("npm"),
    text(
        "npm.pass-with-no-tests-config",
        WEAK,
        SUP,
        Pat::HasAny(&["\"passWithNoTests\" : true", "passWithNoTests : true"]),
        "`passWithNoTests` makes a run that found no tests a pass",
    )
    .tool("jest"),
    text(
        "npm.test-path-ignore",
        SKIP,
        EVA,
        Pat::Word("testPathIgnorePatterns"),
        "`testPathIgnorePatterns` removes test files from the run",
    )
    .tool("jest")
    .waiver(Waiver::Reason),
    text(
        "npm.eslint-quiet",
        LS,
        EVA,
        Pat::Flag("--quiet"),
        "`eslint --quiet` hides every warning",
    )
    .tool("eslint")
    .requires(&["eslint"]),
];

const GOLANGCI: &[Rule] = &[
    code(
        "golangci.disable",
        LS,
        SUP,
        Pat::Lead(&["disable:"]),
        "a `disable:` list switches linters off",
    )
    .tool("golangci-lint"),
    code(
        "golangci.exit-code",
        CFG,
        FRAUD,
        Pat::Lead(&["issues-exit-code: 0"]),
        "`issues-exit-code: 0` makes a run with findings exit successfully",
    )
    .tool("golangci-lint"),
    code(
        "golangci.exclude",
        CFG,
        EVA,
        Pat::Lead(&[
            "exclude-rules:",
            "exclude:",
            "exclude-dirs:",
            "exclude-files:",
            "skip-dirs:",
            "skip-files:",
            "exclusions:",
        ]),
        "excluded files or rules are never linted",
    )
    .tool("golangci-lint"),
    code(
        "golangci.tests-false",
        CFG,
        EVA,
        Pat::Lead(&["tests: false"]),
        "`tests: false` stops test files from being linted",
    )
    .tool("golangci-lint"),
    code(
        "golangci.new-only",
        CFG,
        EVA,
        Pat::Lead(&["new: true", "new-from-rev:", "new-from-patch:"]),
        "reporting only new issues hides every existing one",
    )
    .tool("golangci-lint"),
    code(
        "golangci.disable-all",
        CFG,
        NOTE,
        Pat::Lead(&["disable-all: true", "default: none"]),
        "starts from no linters; fine when followed by an explicit `enable` list, recorded",
    )
    .tool("golangci-lint"),
];

const MSBUILD: &[Rule] = &[
    code(
        "msb.nowarn",
        LS,
        SUP,
        Pat::Has("<NoWarn>"),
        "`<NoWarn>` silences compiler or analyzer warnings project-wide",
    )
    .tool("msbuild"),
    code(
        "msb.warnaserror-off",
        CFG,
        SUP,
        Pat::HasAny(&[
            "<TreatWarningsAsErrors>false",
            "<CodeAnalysisTreatWarningsAsErrors>false",
            "<MSBuildTreatWarningsAsErrors>false",
        ]),
        "warnings no longer fail the build",
    )
    .tool("msbuild")
    .ci(),
    code(
        "msb.warning-level-0",
        CFG,
        FRAUD,
        Pat::Has("<WarningLevel>0<"),
        "`<WarningLevel>0` disables every compiler warning",
    )
    .tool("msbuild"),
    code(
        "msb.analyzers-off",
        CFG,
        SUP,
        Pat::HasAny(&[
            "<RunAnalyzers>false",
            "<RunAnalyzersDuringBuild>false",
            "<EnableNETAnalyzers>false",
            "<AnalysisLevel>none",
            "<AnalysisMode>None",
            "<EnforceCodeStyleInBuild>false",
        ]),
        "the setting turns code analyzers off",
    )
    .tool("roslyn")
    .ci(),
    code(
        "msb.nullable-disable",
        TYPE,
        EVA,
        Pat::Has("<Nullable>disable"),
        "`<Nullable>disable` turns null-safety analysis off project-wide",
    )
    .tool("roslyn")
    .ci(),
    code(
        "msb.warnings-not-as-errors",
        CFG,
        SUP,
        Pat::Has("<WarningsNotAsErrors>"),
        "`<WarningsNotAsErrors>` demotes listed errors back to warnings",
    )
    .tool("msbuild"),
    text(
        "msb.continue-on-error",
        WEAK,
        SUP,
        Pat::HasAny(&[
            "ContinueOnError=\"true\"",
            "ContinueOnError=\"WarnAndContinue\"",
        ]),
        "`ContinueOnError` lets a failing task pass the build",
    )
    .tool("msbuild")
    .ci(),
    text(
        "msb.ruleset-none",
        LS,
        SUP,
        Pat::Has("Action=\"None\""),
        "a rule set entry with `Action=\"None\"` switches the rule off",
    )
    .tool("roslyn"),
    code(
        "msb.istest-false",
        SKIP,
        EVA,
        Pat::Has("<IsTestProject>false"),
        "`<IsTestProject>false` keeps `dotnet test` from running the project",
    )
    .tool("msbuild"),
];

const EDITORCONFIG: &[Rule] = &[
    check(
        "ec.diagnostic-off",
        LS,
        SUP,
        dotnet_diagnostic_off,
        "a `dotnet_diagnostic.*.severity` of none/silent/suggestion switches the analyzer rule off",
    )
    .tool("roslyn"),
    code(
        "ec.analyzers-off",
        CFG,
        FRAUD,
        Pat::Seq(&["dotnet_analyzer_diagnostic", "severity", "= none"]),
        "`dotnet_analyzer_diagnostic.severity = none` switches analyzers off wholesale",
    )
    .tool("roslyn"),
    code(
        "ec.generated-code",
        CFG,
        EVA,
        Pat::Lead(&["generated_code = true"]),
        "files marked generated are skipped by analyzers",
    )
    .tool("roslyn"),
];

const GRADLE: &[Rule] = &[
    code(
        "gradle.ignore-failures",
        WEAK,
        FRAUD,
        Pat::HasAny(&[
            "gnoreFailures = true",
            "gnoreFailures true",
            "gnoreFailures.set(true)",
        ]),
        "`ignoreFailures = true` lets a failing test or lint task pass the build",
    )
    .tool("gradle")
    .ci(),
    check(
        "gradle.test-disabled",
        SKIP,
        FRAUD,
        gradle_test_disabled,
        "`enabled = false` on the test task means the tests never run",
    )
    .tool("gradle"),
    text(
        "gradle.exclude-tests",
        SKIP,
        SUP,
        Pat::HasAny(&[
            "excludeTestsMatching",
            "exclude '**/*Test",
            "exclude(\"**/*Test",
            "exclude \"**/*Test",
        ]),
        "a test filter removes tests from the run",
    )
    .tool("gradle")
    .waiver(Waiver::Reason),
    code(
        "gradle.lint-abort-off",
        CFG,
        SUP,
        Pat::HasAny(&[
            "abortOnError false",
            "abortOnError = false",
            "isAbortOnError = false",
            "checkReleaseBuilds false",
            "warningsAsErrors false",
        ]),
        "Android lint findings no longer fail the build",
    )
    .tool("android-lint"),
    code(
        "gradle.suppress-warnings",
        LS,
        SUP,
        Pat::HasAny(&[
            "suppressWarnings = true",
            "suppressWarnings.set(true)",
            "allWarningsAsErrors = false",
        ]),
        "compiler warnings are suppressed or no longer fail the build",
    )
    .tool("kotlinc"),
    text(
        "gradle.nowarn",
        LS,
        SUP,
        Pat::HasAny(&["-nowarn", "-Xlint:none"]),
        "`-nowarn`/`-Xlint:none` silences javac warnings",
    )
    .tool("javac"),
    code(
        "gradle.fail-on-error-off",
        CFG,
        EVA,
        Pat::Has("failOnError = false"),
        "`failOnError = false` lets a failing task pass",
    )
    .tool("gradle"),
];

const MAVEN: &[Rule] = &[
    code(
        "mvn.skip-tests",
        SKIP,
        FRAUD,
        Pat::HasAny(&["<skipTests>true", "<maven.test.skip>true"]),
        "the property skips the test phase",
    )
    .tool("maven"),
    code(
        "mvn.failure-ignore",
        WEAK,
        FRAUD,
        Pat::HasAny(&["<testFailureIgnore>true", "<maven.test.failure.ignore>true"]),
        "test failures no longer fail the build",
    )
    .tool("maven"),
    code(
        "mvn.skip-its",
        SKIP,
        SUP,
        Pat::Has("<skipITs>true"),
        "integration tests are skipped",
    )
    .tool("maven"),
    code(
        "mvn.linter-skip",
        CFG,
        SUP,
        Pat::HasAny(&[
            "<checkstyle.skip>true",
            "<pmd.skip>true",
            "<spotbugs.skip>true",
            "<jacoco.skip>true",
            "<cpd.skip>true",
            "<enforcer.skip>true",
        ]),
        "the property skips a linter or coverage gate",
    )
    .tool("maven"),
    code(
        "mvn.fail-off",
        CFG,
        SUP,
        Pat::HasAny(&[
            "<failOnViolation>false",
            "<failsOnError>false",
            "<failOnError>false",
        ]),
        "the linter's findings no longer fail the build",
    )
    .tool("maven"),
    code(
        "mvn.plugin-skip",
        CFG,
        EVA,
        Pat::Has("<skip>true</skip>"),
        "a plugin execution is skipped",
    )
    .tool("maven"),
    text(
        "mvn.exclude-tests",
        SKIP,
        SUP,
        Pat::Seq(&["<exclude>", "Test"]),
        "an `<exclude>` pattern removes tests from the run",
    )
    .tool("surefire")
    .waiver(Waiver::Reason),
    text(
        "mvn.suppress-all",
        LS,
        FRAUD,
        Pat::Has("checks=\".*\""),
        "a suppression matching every check switches Checkstyle off",
    )
    .tool("checkstyle"),
    text(
        "mvn.suppress",
        LS,
        SUP,
        Pat::Has("<suppress "),
        "a Checkstyle suppression switches a check off for the matched files",
    )
    .tool("checkstyle")
    .except(&["checks=\".*\""]),
];

const RUBOCOP_CONFIG: &[Rule] = &[
    code(
        "rbcfg.enabled-false",
        CFG,
        SUP,
        Pat::Lead(&["Enabled: false"]),
        "a cop set `Enabled: false` is switched off",
    )
    .tool("rubocop"),
    code(
        "rbcfg.disabled-by-default",
        CFG,
        FRAUD,
        Pat::Lead(&["DisabledByDefault: true"]),
        "`DisabledByDefault: true` switches every cop off",
    )
    .tool("rubocop"),
    code(
        "rbcfg.exclude",
        CFG,
        EVA,
        Pat::Lead(&["Exclude:"]),
        "excluded files are never inspected",
    )
    .tool("rubocop"),
    text(
        "rbcfg.todo",
        CFG,
        NOTE,
        Pat::Has(".rubocop_todo.yml"),
        "inherits a generated to-do list of accepted offenses; recorded",
    )
    .tool("rubocop"),
    check(
        "rbcfg.max-raised",
        CFG,
        EVA,
        rubocop_max_raised,
        "a metric `Max` raised to 200 or more silences the cop it bounds",
    )
    .tool("rubocop"),
];

const PHP_LINT_CONFIG: &[Rule] = &[
    code(
        "phpcfg.level-0",
        CFG,
        SUP,
        Pat::Lead(&["level: 0"]),
        "PHPStan level 0 checks almost nothing",
    )
    .tool("phpstan"),
    code(
        "phpcfg.ignore-errors",
        CFG,
        SUP,
        Pat::Lead(&["ignoreErrors:"]),
        "an `ignoreErrors` list silences analyzer findings",
    )
    .tool("phpstan"),
    code(
        "phpcfg.unmatched-off",
        CFG,
        EVA,
        Pat::Lead(&["reportUnmatchedIgnoredErrors: false"]),
        "stale ignores are no longer reported, so suppressions accumulate",
    )
    .tool("phpstan"),
    text(
        "phpcfg.psalm-level",
        CFG,
        SUP,
        Pat::HasAny(&["errorLevel=\"8\"", "errorLevel=\"7\""]),
        "Psalm error level 7-8 reports almost nothing",
    )
    .tool("psalm"),
    text(
        "phpcfg.psalm-suppress",
        LS,
        SUP,
        Pat::Has("errorLevel=\"suppress\""),
        "an issue handler set to `suppress` silences the issue",
    )
    .tool("psalm"),
    text(
        "phpcfg.phpcs-exclude",
        LS,
        SUP,
        Pat::HasAny(&["<exclude name=", "<severity>0</severity>"]),
        "the ruleset switches a sniff off",
    )
    .tool("phpcs"),
    text(
        "phpcfg.phpunit-exclude",
        SKIP,
        EVA,
        Pat::Seq(&["<exclude>", "test"]),
        "an `<exclude>` in the test suite removes tests from the run",
    )
    .tool("phpunit")
    .ci(),
    text(
        "phpcfg.phpunit-lax",
        WEAK,
        EVA,
        Pat::HasAny(&[
            "beStrictAboutTestsThatDoNotTestAnything=\"false\"",
            "failOnRisky=\"false\"",
            "failOnWarning=\"false\"",
        ]),
        "tests that assert nothing, or warn, no longer fail the run",
    )
    .tool("phpunit"),
];

const SWIFTLINT_CONFIG: &[Rule] = &[
    code(
        "swcfg.disabled-rules",
        CFG,
        SUP,
        Pat::Lead(&["disabled_rules:"]),
        "a `disabled_rules` list switches rules off",
    )
    .tool("swiftlint"),
    code(
        "swcfg.excluded",
        CFG,
        EVA,
        Pat::Lead(&["excluded:"]),
        "excluded paths are never linted",
    )
    .tool("swiftlint"),
    code(
        "swcfg.allow-zero",
        CFG,
        EVA,
        Pat::Lead(&["allow_zero_lintable_files: true"]),
        "a run that linted nothing passes",
    )
    .tool("swiftlint"),
];

const DART_ANALYSIS: &[Rule] = &[
    check(
        "dartcfg.error-ignore",
        LS,
        SUP,
        dart_analysis_ignore,
        "an analyzer error set to `ignore` is silenced project-wide",
    )
    .tool("dart-analyzer"),
    check(
        "dartcfg.rule-false",
        CFG,
        SUP,
        dart_rule_false,
        "a lint rule or strict mode set to `false` is switched off",
    )
    .tool("dart-analyzer"),
    code(
        "dartcfg.exclude",
        CFG,
        EVA,
        Pat::Lead(&["exclude:"]),
        "excluded paths are never analyzed",
    )
    .tool("dart-analyzer"),
];

const CMAKE: &[Rule] = &[
    text(
        "cmake.warning-as-error-off",
        CFG,
        SUP,
        Pat::Seq(&["CMAKE_COMPILE_WARNING_AS_ERROR", "OFF"]),
        "warnings no longer fail the build",
    )
    .tool("cmake"),
    text(
        "cmake.test-disabled",
        SKIP,
        SUP,
        Pat::SeqWord(&["set_tests_properties", "DISABLED"]),
        "a `DISABLED` test property keeps the test from running",
    )
    .tool("ctest"),
    text(
        "cmake.will-fail",
        WEAK,
        SUP,
        Pat::SeqWord(&["set_tests_properties", "WILL_FAIL"]),
        "`WILL_FAIL` inverts the test: it passes by failing",
    )
    .tool("ctest"),
    text(
        "cmake.skip-return-code",
        SKIP,
        EVA,
        Pat::Has("SKIP_RETURN_CODE"),
        "a skip return code lets the test report itself as skipped",
    )
    .tool("ctest"),
    code(
        "cmake.if-false",
        SKIP,
        EVA,
        Pat::HasAny(&["if ( FALSE )", "if ( 0 )"]),
        "an always-false `if` keeps the block from configuring",
    )
    .tool("cmake")
    .ci(),
];

const CLANG_TIDY_CONFIG: &[Rule] = &[
    check(
        "ctcfg.all-off",
        CFG,
        FRAUD,
        clang_tidy_all_off,
        "`Checks: '-*'` with nothing enabled after it runs no checks",
    )
    .tool("clang-tidy"),
    check(
        "ctcfg.check-off",
        LS,
        SUP,
        clang_tidy_check_off,
        "a `-check` entry switches a clang-tidy check off",
    )
    .tool("clang-tidy"),
    code(
        "ctcfg.warnings-as-errors-empty",
        CFG,
        NOTE,
        Pat::Lead(&["WarningsAsErrors: ''", "WarningsAsErrors: \"\""]),
        "no clang-tidy finding fails the build; the default, recorded",
    )
    .tool("clang-tidy"),
];

// ---------------------------------------------------------------------------
// Structural checks
// ---------------------------------------------------------------------------

fn yes(cond: bool) -> CheckHit {
    if cond {
        CheckHit::Yes
    } else {
        CheckHit::No
    }
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn code_of<'d>(doc: &'d Doc<'_>, i: usize) -> &'d str {
    doc.lines.get(i).map_or("", |l| l.code.as_str())
}

fn text_of<'d>(doc: &'d Doc<'_>, i: usize) -> &'d str {
    doc.lines.get(i).map_or("", |l| l.text.as_str())
}

/// The nearest line after `i` with code on it.
fn next_code_line(doc: &Doc<'_>, i: usize) -> Option<usize> {
    (i + 1..doc.lines.len()).find(|&j| !code_of(doc, j).trim().is_empty())
}

/// The nearest line before `i` with code on it.
fn prev_code_line(doc: &Doc<'_>, i: usize) -> Option<usize> {
    (0..i).rev().find(|&j| !code_of(doc, j).trim().is_empty())
}

/// A word from `words` in `hay`, as [`Pat::Word`].
fn has_word(hay: &str, words: &[&str], ci: bool) -> bool {
    let b = hay.as_bytes();
    words
        .iter()
        .any(|w| find(b, 0, b.len(), w, ci, true).is_some())
}

/// A Rust attribute that opens on a line: its body with whitespace removed
/// (code view), its body with strings kept (text view), and how many lines
/// it spans.
struct RustAttr {
    body: String,
    text: String,
    span: usize,
}

/// Every attribute that opens on line `i`, however many lines it spans
/// (bounded at 64): `#[`, `#![`, and `#` and `[` on different lines.
fn rust_attrs(doc: &Doc<'_>, i: usize) -> Vec<RustAttr> {
    let Some(line) = doc.lines.get(i) else {
        return Vec::new();
    };
    if !line.code.contains('#') {
        return Vec::new();
    }
    let window = doc.lines.get(i..doc.lines.len().min(i + 64)).unwrap_or(&[]);
    let code = window
        .iter()
        .map(|l| l.code.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let text = window
        .iter()
        .map(|l| l.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let cb = code.as_bytes();
    let skip_ws = |mut k: usize| {
        while at(cb, k).is_ascii_whitespace() {
            k += 1;
        }
        k
    };
    let mut out = Vec::new();
    for p in (0..line.code.len()).filter(|&p| at(cb, p) == b'#') {
        let mut k = skip_ws(p + 1);
        if at(cb, k) == b'!' {
            k = skip_ws(k + 1);
        }
        if at(cb, k) != b'[' {
            continue;
        }
        let start = k + 1;
        let mut depth = 1usize;
        let mut j = start;
        while j < cb.len() && depth > 0 {
            match at(cb, j) {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        if depth != 0 {
            continue;
        }
        let inner = start..j.saturating_sub(1);
        out.push(RustAttr {
            body: squash(code.get(inner.clone()).unwrap_or("")),
            text: text.get(inner).unwrap_or("").to_string(),
            span: code.get(p..j).map_or(1, |s| s.matches('\n').count() + 1),
        });
    }
    out
}

/// `cfg` predicates that are never true, plus the made-up names used to
/// mean "never".
const FALSE_CFGS: &[&str] = &[
    "any()",
    "not(all())",
    "FALSE",
    "false",
    "never",
    "disabled",
    "disable",
    "skip",
    "ignore",
    "off",
    "none",
    "broken",
];

fn rust_always_false_cfg(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(rust_attrs(doc, i).iter().any(|a| {
        a.body
            .strip_prefix("cfg(")
            .and_then(|r| r.strip_suffix(')'))
            .is_some_and(|inner| FALSE_CFGS.contains(&inner))
    }))
}

fn rust_cfg_attr_ignore(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(rust_attrs(doc, i).iter().any(|a| {
        a.body.starts_with("cfg_attr(")
            && (a.body.contains(",ignore)")
                || a.body.contains(",ignore=")
                || a.body.contains(",ignore,"))
    }))
}

fn rust_should_panic_bare(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(rust_attrs(doc, i).iter().any(|a| a.body == "should_panic"))
}

fn rust_lint_attr_long(doc: &Doc<'_>, i: usize) -> CheckHit {
    let silencing = |b: &str| {
        b.starts_with("allow(")
            || b.starts_with("expect(")
            || (b.starts_with("cfg_attr(") && (b.contains(",allow(") || b.contains(",expect(")))
    };
    let mut verdict = CheckHit::No;
    for a in rust_attrs(doc, i) {
        if a.span > 4 && silencing(&a.body) {
            let reasoned = a.text.find("reason").is_some_and(|k| {
                a.text
                    .get(k + "reason".len()..)
                    .map(str::trim_start)
                    .and_then(|r| r.strip_prefix('='))
                    .map(str::trim_start)
                    .and_then(|r| r.strip_prefix('"'))
                    .is_some_and(|r| has_words(r.split('"').next().unwrap_or("")))
            });
            if !reasoned {
                return CheckHit::Yes;
            }
            verdict = CheckHit::Justified;
        }
    }
    verdict
}

/// Calls whose `Result` reports that a write or a filesystem change did not
/// happen. Discarding one leaves the program continuing as if it had.
const IO_MUTATIONS: &[&str] = &[
    "write_all(",
    ".write(",
    ".flush(",
    "sync_all(",
    "sync_data(",
    "set_len(",
    "fs::write(",
    "remove_file(",
    "remove_dir(",
    "remove_dir_all(",
    "create_dir(",
    "create_dir_all(",
    "fs::rename(",
    "fs::copy(",
    "set_permissions(",
    "hard_link(",
    ".persist(",
];

/// A Cargo build script: it prints `cargo:` directives or reads `OUT_DIR`.
/// What it writes lands in `OUT_DIR` and is read by an `include!` in the
/// crate, so a write it failed to make surfaces as a compile error at that
/// read — the discard can delay the failure, not hide it — and its
/// environment probes are optional by design.
fn rust_build_script(doc: &Doc<'_>) -> bool {
    doc.lines
        .iter()
        .any(|l| l.raw.contains("cargo:") || l.raw.contains("OUT_DIR"))
}

/// Whether discarding the value on line `i` throws away an I/O error that
/// matters. Discards in general are idiomatic Rust — a send on a channel
/// whose receiver may be gone, a `join` at shutdown, an optional environment
/// probe — and are not reported; this crate denies `let _ =` by its own lint
/// table, but the data path reads general Rust, not this crate. What is
/// reported is the discard that loses data silently: a write, a flush, a
/// sync or a filesystem change, outside a build script.
fn discards_io_error(doc: &Doc<'_>, line: &str) -> bool {
    IO_MUTATIONS.iter().any(|m| line.contains(m)) && !rust_build_script(doc)
}

/// `let _ = <a write or filesystem change>`. See [`discards_io_error`].
fn rust_let_underscore_io(doc: &Doc<'_>, i: usize) -> CheckHit {
    let code = code_of(doc, i);
    let Some(at) = code.find("let _ =") else {
        return CheckHit::No;
    };
    yes(discards_io_error(doc, code.get(at..).unwrap_or("")))
}

/// A statement `<a write or filesystem change>.ok();`. See
/// [`discards_io_error`].
fn rust_ok_discard(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = squash(code_of(doc, i));
    yes(t.ends_with(".ok();")
        && !t.contains('=')
        && !t.starts_with("return")
        && !t.starts_with("let")
        && discards_io_error(doc, &t))
}

fn py_ellipsis_body(doc: &Doc<'_>, i: usize) -> CheckHit {
    if code_of(doc, i).trim() != "..." {
        return CheckHit::No;
    }
    // Skip a docstring between the `def` and the `...`.
    let mut j = prev_code_line(doc, i);
    while let Some(k) = j {
        if code_of(doc, k)
            .trim()
            .chars()
            .all(|c| c == '"' || c == '\'')
        {
            j = prev_code_line(doc, k);
        } else {
            break;
        }
    }
    let Some(def) = j else {
        return CheckHit::No;
    };
    let head = code_of(doc, def).trim();
    if !((head.starts_with("def ") || head.starts_with("async def ")) && head.ends_with(':')) {
        return CheckHit::No;
    }
    // `@overload` and `@abstractmethod` bodies are declarations, and so is a
    // method of a `Protocol`.
    let decorated = (0..def)
        .rev()
        .map(|k| code_of(doc, k).trim())
        .take_while(|t| t.starts_with('@'))
        .any(|t| t.contains("overload") || t.contains("abstractmethod"));
    let indent = code_of(doc, def).len() - code_of(doc, def).trim_start().len();
    let protocol = (0..def).rev().take(200).any(|k| {
        let c = code_of(doc, k);
        let ind = c.len() - c.trim_start().len();
        ind < indent && c.trim_start().starts_with("class ") && c.contains("Protocol")
    });
    yes(!decorated && !protocol)
}

fn empty_catch(doc: &Doc<'_>, i: usize) -> CheckHit {
    let code = code_of(doc, i);
    let b = code.as_bytes();
    let skip = |mut k: usize| {
        while at(b, k).is_ascii_whitespace() {
            k += 1;
        }
        k
    };
    let mut from = 0;
    while let Some((s, e)) = find(b, from, b.len(), "catch", false, true) {
        from = s + 1;
        let mut k = skip(e);
        if at(b, k) == b'(' {
            let mut depth = 0usize;
            while k < b.len() {
                match at(b, k) {
                    b'(' => depth += 1,
                    b')' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                k += 1;
            }
            k = skip(k + 1);
        }
        if at(b, k) != b'{' {
            continue;
        }
        k = skip(k + 1);
        if at(b, k) == b'}' {
            return CheckHit::Yes;
        }
        if k >= b.len()
            && next_code_line(doc, i).is_some_and(|j| code_of(doc, j).trim().starts_with('}'))
        {
            return CheckHit::Yes;
        }
    }
    CheckHit::No
}

fn js_eslint_inline_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    let Some(line) = doc.lines.get(i) else {
        return CheckHit::No;
    };
    let b = line.raw.as_bytes();
    for (p, end) in directive_starts(line) {
        if !(starts_with_at(b, p, "eslint") && matches!(at(b, p + 6), b' ' | b'\t')) {
            continue;
        }
        let body = line.raw.get(p + 6..end).unwrap_or("");
        for (k, _) in body.match_indices(':') {
            let value = body.get(k + 1..).unwrap_or("").trim_start();
            let value = value.strip_prefix('[').map_or(value, str::trim_start);
            let off = ["\"off\"", "'off'", "off"].iter().any(|v| {
                value
                    .strip_prefix(v)
                    .is_some_and(|r| !r.starts_with(|c: char| c.is_alphanumeric()))
            });
            let zero = value
                .strip_prefix('0')
                .is_some_and(|r| !r.starts_with(|c: char| c.is_ascii_digit() || c == '.'));
            if off || zero {
                return CheckHit::Yes;
            }
        }
    }
    CheckHit::No
}

fn hs_ghc_options_suppress(doc: &Doc<'_>, i: usize) -> CheckHit {
    let Some(line) = doc.lines.get(i) else {
        return CheckHit::No;
    };
    yes(line.comments.iter().any(|s| {
        let body = line.raw.get(s.body..s.end).unwrap_or("");
        body.find("OPTIONS_GHC").is_some_and(|k| {
            body.get(k..).unwrap_or("").split_whitespace().any(|t| {
                t == "-w" || t == "-Wwarn" || t.starts_with("-Wno-") || t.starts_with("-fno-warn-")
            })
        })
    }))
}

fn php_error_suppression(doc: &Doc<'_>, i: usize) -> CheckHit {
    let b = code_of(doc, i).as_bytes();
    yes((0..b.len()).any(|k| {
        at(b, k) == b'@'
            && !is_ident(at_prev(b, k))
            && (at(b, k + 1) == b'$' || at(b, k + 1).is_ascii_alphabetic() || at(b, k + 1) == b'_')
    }))
}

fn zig_discard_call(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = code_of(doc, i).trim();
    let rest = t
        .strip_prefix('_')
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix('='));
    yes(rest.is_some_and(|r| !r.starts_with('=') && r.contains('(') && r.trim_end().ends_with(';')))
}

fn zig_unreachable_body(doc: &Doc<'_>, i: usize) -> CheckHit {
    if code_of(doc, i).trim() != "unreachable;" {
        return CheckHit::No;
    }
    yes(prev_code_line(doc, i).is_some_and(|j| {
        let t = code_of(doc, j).trim();
        t.contains("fn ") && t.ends_with('{')
    }))
}

fn nim_discard_body(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = code_of(doc, i).trim_start();
    let routine = ["proc ", "func ", "method ", "iterator ", "converter "]
        .iter()
        .any(|k| t.starts_with(k));
    yes(routine && squash(t).ends_with("=discard"))
}

/// Targets whose recipe is the gate.
const GATE_TARGETS: &[&str] = &[
    "test",
    "tests",
    "check",
    "lint",
    "gate",
    "verify",
    "audit",
    "clippy",
    "typecheck",
    "ci",
];

fn make_dash_gate(doc: &Doc<'_>, i: usize) -> CheckHit {
    let Some(line) = doc.lines.get(i) else {
        return CheckHit::No;
    };
    let Some(recipe) = line.raw.strip_prefix('\t') else {
        return CheckHit::No;
    };
    let recipe = recipe.trim_start();
    let prefix: String = recipe
        .chars()
        .take_while(|c| matches!(c, '@' | '-' | '+'))
        .collect();
    yes(prefix.contains('-') && has_word(&line.text, GATE_CMDS, false))
}

fn noop_command(cmd: &str) -> bool {
    let c = cmd.trim().trim_start_matches(['@', '-', '+']).trim();
    c == "true" || c == ":" || c == "exit 0" || c.starts_with("echo ") || c == "echo"
}

fn make_noop_gate_recipe(doc: &Doc<'_>, i: usize) -> CheckHit {
    let Some(line) = doc.lines.get(i) else {
        return CheckHit::No;
    };
    let code = line.code.as_str();
    if code.starts_with(char::is_whitespace) {
        return CheckHit::No;
    }
    let Some((target, rest)) = code.split_once(':') else {
        return CheckHit::No;
    };
    if rest.starts_with('=') || !GATE_TARGETS.contains(&target.trim()) {
        return CheckHit::No;
    }
    let (deps, inline) = match rest.split_once(';') {
        Some((d, r)) => (d.trim(), Some(r.trim())),
        None => (rest.trim(), None),
    };
    let mut recipe: Vec<&str> = inline.into_iter().filter(|r| !r.is_empty()).collect();
    for l in doc.lines.iter().skip(i + 1) {
        if let Some(cmd) = l.raw.strip_prefix('\t') {
            if !l.code.trim().is_empty() {
                recipe.push(cmd);
            }
        } else if l.raw.trim().is_empty() || l.is_comment_only() {
            continue;
        } else {
            break;
        }
    }
    yes(if recipe.is_empty() {
        deps.is_empty()
    } else {
        recipe.iter().all(|c| noop_command(c))
    })
}

fn npm_noop_script(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = text_of(doc, i).trim_start();
    let Some(rest) = t.strip_prefix('"') else {
        return CheckHit::No;
    };
    let Some((key, after)) = rest.split_once('"') else {
        return CheckHit::No;
    };
    let gate = ["test", "lint", "typecheck", "check", "verify"]
        .iter()
        .any(|g| key == *g || key.starts_with(&format!("{g}:")));
    if !gate {
        return CheckHit::No;
    }
    let Some(value) = after.trim_start().strip_prefix(':').map(str::trim_start) else {
        return CheckHit::No;
    };
    let Some(value) = value.strip_prefix('"') else {
        return CheckHit::No;
    };
    // The value ends at the first quote not escaped by a backslash.
    let vb = value.as_bytes();
    let end = (0..vb.len())
        .find(|&k| at(vb, k) == b'"' && at_prev(vb, k) != b'\\')
        .unwrap_or(vb.len());
    let cmd = value.get(..end).unwrap_or("").replace("\\\"", "\"");
    let chained = cmd.contains("&&") || cmd.contains("||") || cmd.contains(';');
    yes(cmd.trim().is_empty() || (!chained && noop_command(&cmd)))
}

/// The TOML table header governing line `i`, whitespace removed.
fn toml_header(doc: &Doc<'_>, i: usize) -> Option<String> {
    (0..i)
        .rev()
        .map(|j| code_of(doc, j).trim())
        .find(|t| t.starts_with('['))
        .map(squash)
}

fn cargo_test_harness_false(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(squash(code_of(doc, i)) == "harness=false"
        && toml_header(doc, i).as_deref() == Some("[[test]]"))
}

fn cargo_test_required_features(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(squash(code_of(doc, i)).starts_with("required-features=")
        && toml_header(doc, i).as_deref() == Some("[[test]]"))
}

fn cargo_test_profile_weakened(doc: &Doc<'_>, i: usize) -> CheckHit {
    let line = squash(code_of(doc, i));
    let weakened = line == "debug-assertions=false" || line == "overflow-checks=false";
    let header = toml_header(doc, i);
    yes(weakened && matches!(header.as_deref(), Some("[profile.test]" | "[profile.dev]")))
}

fn cargo_alias_shadows_gate(doc: &Doc<'_>, i: usize) -> CheckHit {
    const GATES: &[&str] = &["clippy", "test", "check", "fmt", "build", "t", "c"];
    let line = squash(code_of(doc, i));
    let Some((key, _)) = line.split_once('=') else {
        return CheckHit::No;
    };
    let in_alias = toml_header(doc, i).as_deref() == Some("[alias]") && GATES.contains(&key);
    let dotted = key
        .strip_prefix("alias.")
        .is_some_and(|k| GATES.contains(&k));
    yes(in_alias || dotted)
}

fn cargo_lints_lowered(doc: &Doc<'_>, i: usize) -> CheckHit {
    let line = squash(code_of(doc, i));
    let lowers = ["=\"allow\"", "=\"warn\"", "='allow'", "='warn'"]
        .iter()
        .any(|n| line.contains(n));
    if !lowers || line.starts_with('[') {
        return CheckHit::No;
    }
    let names_lints = |s: &str| {
        s.trim_matches(['[', ']'])
            .split('.')
            .any(|seg| seg.trim_matches('"') == "lints")
    };
    let key = line.split('=').next().unwrap_or("");
    yes(toml_header(doc, i).is_some_and(|h| names_lints(&h)) || names_lints(key))
}

/// Clippy thresholds and their defaults; raising one silences its lint.
const CLIPPY_DEFAULTS: &[(&str, u64)] = &[
    ("too-many-arguments-threshold", 7),
    ("too-many-lines-threshold", 100),
    ("cognitive-complexity-threshold", 25),
    ("type-complexity-threshold", 250),
    ("large-error-threshold", 128),
    ("enum-variant-size-threshold", 200),
    ("too-large-for-stack", 200),
    ("pass-by-value-size-limit", 256),
    ("max-fn-params-bools", 3),
    ("max-struct-bools", 3),
    ("max-trait-bounds", 3),
    ("single-char-binding-names-threshold", 4),
    ("vec-box-size-threshold", 4096),
    ("stack-size-threshold", 512_000),
    ("array-size-threshold", 16_384),
    ("literal-representation-threshold", 16_384),
    ("enum-variant-name-threshold", 3),
    ("struct-field-name-threshold", 3),
    ("future-size-threshold", 16_384),
    ("unnecessary-box-size", 128),
];

fn clippy_threshold_raised(doc: &Doc<'_>, i: usize) -> CheckHit {
    let line = squash(code_of(doc, i));
    let Some((key, value)) = line.split_once('=') else {
        return CheckHit::No;
    };
    let Ok(value) = value.replace('_', "").parse::<u64>() else {
        return CheckHit::No;
    };
    yes(CLIPPY_DEFAULTS
        .iter()
        .any(|&(k, default)| k == key && value >= default.saturating_mul(3)))
}

fn pyright_report_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = text_of(doc, i).trim_start().trim_start_matches(['"', '\'']);
    let Some(rest) = t.strip_prefix("report") else {
        return CheckHit::No;
    };
    if !rest.starts_with(|c: char| c.is_ascii_uppercase()) {
        return CheckHit::No;
    }
    let Some(k) = rest.find(['=', ':']) else {
        return CheckHit::No;
    };
    let value = rest
        .get(k + 1..)
        .unwrap_or("")
        .trim()
        .trim_start_matches(['"', '\''])
        .to_ascii_lowercase();
    yes(value.starts_with("false") || value.starts_with("none"))
}

fn fail_under_zero(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = text_of(doc, i);
    for key in ["fail_under", "fail-under"] {
        let mut from = 0;
        while let Some(k) = t.get(from..).and_then(|s| s.find(key)).map(|k| k + from) {
            from = k + key.len();
            let rest = t.get(from..).unwrap_or("").trim_start();
            let rest = rest.strip_prefix(['=', ':']).unwrap_or(rest).trim_start();
            let number: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-'))
                .collect();
            if number.parse::<f64>().is_ok_and(|v| v <= 0.0) {
                return CheckHit::Yes;
            }
        }
    }
    CheckHit::No
}

fn config_rule_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    const NOT_RULES: &[&str] = &[
        "ecmaVersion",
        "version",
        "indent",
        "tabWidth",
        "printWidth",
        "root",
    ];
    let t = text_of(doc, i);
    for (k, _) in t.match_indices(':') {
        let key = t
            .get(..k)
            .unwrap_or("")
            .trim_end()
            .trim_end_matches(['"', '\''])
            .rsplit(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '{' | ','))
            .next()
            .unwrap_or("");
        if key.is_empty() || NOT_RULES.contains(&key) {
            continue;
        }
        let value = t.get(k + 1..).unwrap_or("").trim_start();
        let value = value.strip_prefix('[').map_or(value, str::trim_start);
        let off = ["\"off\"", "'off'", "off"].iter().any(|v| {
            value
                .strip_prefix(v)
                .is_some_and(|r| !r.starts_with(|c: char| c.is_alphanumeric()))
        });
        let zero = value
            .strip_prefix('0')
            .is_some_and(|r| !r.starts_with(|c: char| c.is_ascii_digit() || c == '.'));
        if off || zero {
            return CheckHit::Yes;
        }
    }
    CheckHit::No
}

fn biome_linter_disabled(doc: &Doc<'_>, i: usize) -> CheckHit {
    let b = text_of(doc, i).as_bytes();
    if find(b, 0, b.len(), "\"enabled\" : false", false, false).is_none() {
        return CheckHit::No;
    }
    yes((i.saturating_sub(3)..=i).any(|j| text_of(doc, j).contains("\"linter\"")))
}

fn dotnet_diagnostic_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    let t = code_of(doc, i).trim();
    if !t.starts_with("dotnet_diagnostic.") {
        return CheckHit::No;
    }
    let value = t
        .split_once('=')
        .map_or("", |(_, v)| v)
        .trim()
        .to_ascii_lowercase();
    yes(matches!(value.as_str(), "none" | "silent" | "suggestion"))
}

fn gradle_test_disabled(doc: &Doc<'_>, i: usize) -> CheckHit {
    let line = squash(code_of(doc, i));
    let off = [
        "enabled=false",
        "isEnabled=false",
        "enabledfalse",
        "enabled(false)",
        "enabled.set(false)",
    ]
    .iter()
    .any(|n| line.contains(n));
    if !off {
        return CheckHit::No;
    }
    yes((i.saturating_sub(3)..=i).any(|j| {
        let c = code_of(doc, j);
        has_word(c, &["test"], true) || c.contains("Test>")
    }))
}

fn rubocop_max_raised(doc: &Doc<'_>, i: usize) -> CheckHit {
    let line = squash(code_of(doc, i));
    yes(line
        .strip_prefix("Max:")
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|v| v >= 200))
}

fn yaml_pair(doc: &Doc<'_>, i: usize) -> Option<(String, String)> {
    let t = code_of(doc, i).trim();
    let (key, value) = t.split_once(':')?;
    Some((key.trim().to_string(), value.trim().to_string()))
}

fn dart_analysis_ignore(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(yaml_pair(doc, i).is_some_and(|(k, v)| {
        !k.is_empty()
            && k.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && v == "ignore"
    }))
}

fn dart_rule_false(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(yaml_pair(doc, i).is_some_and(|(k, v)| {
        let rule_like = k.contains('_') || k.starts_with("strict-");
        let stricter_when_false = matches!(k.as_str(), "implicit-casts" | "implicit-dynamic");
        v == "false" && rule_like && !stricter_when_false
    }))
}

fn clang_tidy_entries(doc: &Doc<'_>, i: usize) -> Option<Vec<String>> {
    let t = code_of(doc, i).trim();
    let value = t.strip_prefix("Checks:")?;
    Some(
        value
            .trim()
            .trim_matches(['\'', '"'])
            .split(',')
            .map(|e| e.trim().to_string())
            .filter(|e| !e.is_empty())
            .collect(),
    )
}

fn clang_tidy_all_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(clang_tidy_entries(doc, i)
        .is_some_and(|e| e.iter().any(|x| x == "-*") && e.iter().all(|x| x.starts_with('-'))))
}

fn clang_tidy_check_off(doc: &Doc<'_>, i: usize) -> CheckHit {
    yes(clang_tidy_entries(doc, i)
        .is_some_and(|e| e.iter().any(|x| x.starts_with('-') && x != "-*")))
}

// ---------------------------------------------------------------------------
// Regression corpus
// ---------------------------------------------------------------------------

/// What a case must produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Expect {
    /// At least one hit of this kind above [`Severity::Note`].
    Reject(PolyKind),
    /// At least one [`Severity::Note`] hit of this kind, and nothing above
    /// a note.
    Note(PolyKind),
    /// No hit at all.
    Clean,
}

/// One hand-written regression case.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub lang: Lang,
    pub expect: Expect,
    pub text: &'static str,
}

const fn case(lang: Lang, expect: Expect, text: &'static str) -> Case {
    Case { lang, expect, text }
}

/// The regression corpus: at least two positives, one justified note and
/// one clean negative per [`Lang`], plus negatives that only mention a
/// directive in a string, docstring, heredoc or prose comment.
///
/// This is a regression corpus, not a benchmark: see the module docs.
pub fn regression_cases() -> Vec<Case> {
    use Expect::{Clean, Note, Reject};
    use Lang as L;
    vec![
        // Rust
        case(L::Rust, Reject(SKIP), "#[cfg(any())]\n#[test]\nfn adds() {\n    assert_eq!(add(1, 1), 2);\n}\n"),
        case(L::Rust, Reject(SKIP), "#[cfg(\n    not(\n        all()\n    )\n)]\nmod tests {\n    #[test]\n    fn t() { assert!(f()); }\n}\n"),
        case(L::Rust, Reject(WEAK), "#[test]\n#[should_panic]\nfn rejects() { parse(\"\"); }\n"),
        case(L::Rust, Reject(SKIP), "#[test]\n#[cfg_attr(not(miri), ignore)]\nfn slow() { assert!(run()); }\n"),
        case(L::Rust, Reject(LS), "#[\n\n\n\nallow(\n    clippy::all\n)]\nfn f() -> u8 { 1 }\n"),
        case(L::Rust, Reject(STUB), "fn parse(s: &str) -> u32 {\n    panic! (\"not implemented yet\")\n}\n"),
        case(L::Rust, Reject(ESC), "fn main() {\n    let _ = std::fs::remove_file(\"lock\");\n}\n"),
        case(L::Rust, Note(LS), "#[rustfmt::skip]\nconst M: [u8; 4] = [1, 0,\n                    0, 1];\n"),
        case(L::Rust, Note(ESC), "fn f() {\n    // reason: the lock file may already be gone, and that is the state we want\n    let _ = std::fs::remove_file(\"lock\");\n}\n"),
        case(L::Rust, Note(LS), "#[\n    allow(\n        dead_code,\n        reason = \"kept for the FFI table\"\n    )\n]\nfn ffi() -> u8 { 1 }\n"),
        case(L::Rust, Clean, "fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n\n#[test]\nfn adds() {\n    assert_eq!(add(1, 1), 2);\n}\n"),
        case(L::Rust, Clean, "const DOC: &str = r#\"use #[cfg(any())] to compile out, #[should_panic] for panics, panic! with a todo message\"#;\nfn f() -> &'static str { DOC }\n"),
        case(L::Rust, Clean, "// We never use #[cfg(any())] or let _ = to hide failures.\nfn g() -> u8 { 1 }\n"),
        // Discards that are idiomatic Rust: a send whose receiver may be gone,
        // a join at shutdown, a guard binding, and a build script whose writes
        // are read back by `include!` (a failed write fails that compile).
        case(L::Rust, Clean, "fn notify(tx: &std::sync::mpsc::Sender<u32>, x: u32) {\n    let _ = tx.send(x);\n}\n"),
        case(L::Rust, Clean, "fn stop(handle: std::thread::JoinHandle<()>) {\n    let _ = handle.join();\n    let _guard = LOCK.lock();\n}\n"),
        case(L::Rust, Clean, "use std::{env, fs, path::Path};\n\nfn main() {\n    println!(\"cargo:rerun-if-changed=schema/expected_fields.txt\");\n    let out = env::var(\"OUT_DIR\").unwrap_or_default();\n    fs::write(Path::new(&out).join(\"fields.rs\"), generate()).ok();\n}\n"),
        // The discard that loses data silently stays reported.
        case(L::Rust, Reject(ESC), "fn save(f: &mut std::fs::File, b: &[u8]) {\n    let _ = f.write_all(b);\n}\n"),
        case(L::Rust, Reject(ESC), "fn save(p: &std::path::Path, b: &[u8]) {\n    std::fs::write(p, b).ok();\n}\n"),
        // Python
        case(L::Python, Reject(TYPE), "# mypy: ignore-errors\nimport os\n"),
        case(L::Python, Reject(TYPE), "# pyright: basic\nx: int = 'a'\n"),
        case(L::Python, Reject(LS), "import subprocess\nsubprocess.call(cmd, shell=True)  # nosec\n"),
        case(L::Python, Reject(SKIP), "import pytest\n@pytest.mark.skipif(True, reason=\"flaky\")\ndef test_add():\n    assert add(1, 1) == 2\n"),
        case(L::Python, Reject(WEAK), "@pytest.mark.xfail(strict=False)\ndef test_add():\n    assert add(1, 2) == 3\n"),
        case(L::Python, Reject(STUB), "def area(r):\n    ...\n"),
        case(L::Python, Reject(WEAK), "def test_parse():  # pragma: no cover\n    assert parse('1') == 1\n"),
        case(L::Python, Note(WEAK), "def helper():  # pragma: no cover\n    return 1\n"),
        case(L::Python, Note(LS), "result = yaml.load(data)  # nosec B506 -- reason: input is our own fixture file\n"),
        case(L::Python, Clean, "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n    return a + b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"),
        case(L::Python, Clean, "def lint():\n    \"\"\"Never write # noqa or # mypy: ignore-errors or @pytest.mark.xfail here.\"\"\"\n    return 0\n"),
        case(L::Python, Clean, "# Do not add nosec or pragma: no cover markers to this file.\nx = 1\n"),
        case(L::Python, Clean, "from typing import Protocol\nclass Shape(Protocol):\n    def area(self) -> float:\n        ...\n"),
        case(L::Python, Clean, "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self) -> float:\n        ...\n"),
        case(L::Python, Clean, "from typing import overload\n@overload\ndef parse(s: str) -> int:\n    ...\n@overload\ndef parse(s: bytes) -> int:\n    ...\ndef parse(s):\n    return int(s)\n"),
        // C
        case(L::C, Reject(LS), "int f(int *p) {\n    // cppcheck-suppress nullPointer\n    return *p;\n}\n"),
        case(L::C, Reject(WEAK), "#define NDEBUG\n#include <assert.h>\nint main(void) { assert(add(1, 1) == 3); return 0; }\n"),
        case(L::C, Reject(SKIP), "#if 0\nstatic void test_overflow(void) { assert(add(INT_MAX, 1) < 0); }\n#endif\n"),
        case(L::C, Reject(LS), "#pragma GCC system_header\nint unused_var;\n"),
        case(L::C, Reject(STUB), "int parse(const char *s) {\n    assert(0 && \"not implemented\");\n    return 0;\n}\n"),
        case(L::C, Note(LS), "int f(int *p) {\n    // cppcheck-suppress nullPointer ; reason: p is checked by every caller\n    return *p;\n}\n"),
        case(L::C, Clean, "#include <stdio.h>\nint add(int a, int b) { return a + b; }\n"),
        case(L::C, Clean, "const char *help = \"use // cppcheck-suppress or #pragma GCC system_header\";\n"),
        // C++
        case(L::Cpp, Reject(SKIP), "TEST(Parser, DISABLED_HandlesEmpty) {\n    EXPECT_EQ(parse(\"\"), 0);\n}\n"),
        case(L::Cpp, Reject(SKIP), "TEST(Net, Connects) {\n    GTEST_SKIP() << \"needs network\";\n}\n"),
        case(L::Cpp, Reject(WEAK), "TEST_CASE(\"parses\", \"[!mayfail]\") {\n    REQUIRE(parse(\"1\") == 1);\n}\n"),
        case(L::Cpp, Reject(STUB), "int Parser::next() {\n    throw std::logic_error(\"TODO\");\n}\n"),
        case(L::Cpp, Reject(ESC), "try { save(); } catch (...) {}\n"),
        case(L::Cpp, Note(WEAK), "// LCOV_EXCL_START\nvoid log_debug() { std::puts(\"dbg\"); }\n// LCOV_EXCL_STOP\n"),
        case(L::Cpp, Clean, "#include <vector>\nint sum(const std::vector<int>& v) { int s = 0; for (int x : v) s += x; return s; }\n"),
        case(L::Cpp, Clean, "auto doc = R\"(GTEST_SKIP() and DISABLED_ hide tests; #if 0 too)\";\n"),
        // Objective-C
        case(L::ObjectiveC, Reject(SKIP), "- (void)testLogin {\n    XCTSkip(@\"server down\");\n}\n"),
        case(L::ObjectiveC, Reject(WEAK), "#define NS_BLOCK_ASSERTIONS 1\n#import <Foundation/Foundation.h>\n"),
        case(L::ObjectiveC, Reject(STUB), "- (NSString *)name {\n    @throw [NSException exceptionWithName:@\"Stub\" reason:@\"not implemented\" userInfo:nil];\n}\n"),
        case(L::ObjectiveC, Note(WEAK), "NSLog(@\"debug\"); // LCOV_EXCL_LINE\n"),
        case(L::ObjectiveC, Clean, "@implementation Foo\n- (int)add:(int)a to:(int)b { return a + b; }\n@end\n"),
        // C#
        case(L::CSharp, Reject(LS), "[SuppressMessage(\"Style\", \"IDE0060\")]\npublic void M(int unused) { Use(); }\n"),
        case(L::CSharp, Reject(SKIP), "[Fact(Skip = \"flaky\")]\npublic void Adds() => Assert.Equal(2, Add(1, 1));\n"),
        case(L::CSharp, Reject(STUB), "public int Parse(string s) {\n    throw new NotImplementedException();\n}\n"),
        case(L::CSharp, Reject(TYPE), "#nullable disable\npublic class Model { public string Name; }\n"),
        case(L::CSharp, Reject(ESC), "try { Save(); } catch (Exception) { }\n"),
        case(L::CSharp, Note(LS), "[SuppressMessage(\"Design\", \"CA1031\", Justification = \"The host logs and rethrows every failure\")]\npublic void Run() { Work(); }\n"),
        case(L::CSharp, Clean, "public static class MathX {\n    public static int Add(int a, int b) => a + b;\n}\n"),
        case(L::CSharp, Clean, "var help = @\"Use [Fact(Skip = \"\"x\"\")] and #nullable disable sparingly\";\n"),
        // Java
        case(L::Java, Reject(LS), "@SuppressWarnings(\"unchecked\")\nList<String> xs = (List<String>) raw;\n"),
        case(L::Java, Reject(SKIP), "@Disabled\n@Test\nvoid adds() { assertEquals(2, add(1, 1)); }\n"),
        case(L::Java, Reject(SKIP), "@Test\nvoid adds() {\n    assumeTrue(false);\n    assertEquals(2, add(1, 1));\n}\n"),
        case(L::Java, Reject(LS), "int x = compute(); // NOPMD\n"),
        case(L::Java, Reject(STUB), "public int parse(String s) {\n    // TODO Auto-generated method stub\n    return 0;\n}\n"),
        case(L::Java, Note(LS), "// reason: the raw type comes from a pre-generics API we cannot change\n@SuppressWarnings(\"rawtypes\")\nList xs = legacy();\n"),
        case(L::Java, Clean, "public class MathX {\n    public static int add(int a, int b) { return a + b; }\n}\n"),
        case(L::Java, Clean, "String tip = \"Avoid @Disabled and // NOPMD in tests\";\n"),
        // Kotlin
        case(L::Kotlin, Reject(LS), "@Suppress(\"UNCHECKED_CAST\")\nval xs = raw as List<String>\n"),
        case(L::Kotlin, Reject(STUB), "fun parse(s: String): Int = TODO()\n"),
        case(L::Kotlin, Reject(SKIP), "@Ignore\n@Test\nfun adds() { assertEquals(2, add(1, 1)) }\n"),
        case(L::Kotlin, Note(LS), "@Suppress(\"MagicNumber\") // reason: the port number is fixed by the protocol spec\nval port = 8443\n"),
        case(L::Kotlin, Clean, "fun add(a: Int, b: Int): Int = a + b\n"),
        case(L::Kotlin, Clean, "// TODO() is a stdlib stub; we never ship it\nfun f() = 1\n"),
        // Scala
        case(L::Scala, Reject(STUB), "def parse(s: String): Int = ???\n"),
        case(L::Scala, Reject(LS), "@nowarn(\"msg=deprecated\")\ndef old(): Unit = legacy()\n"),
        case(L::Scala, Reject(SKIP), "ignore(\"parses empty input\") {\n  assert(parse(\"\") == 0)\n}\n"),
        case(L::Scala, Note(LS), "// reason: the deprecated call is the only API on the 2.12 cross-build\n@nowarn(\"cat=deprecation\")\ndef old(): Unit = legacy()\n"),
        case(L::Scala, Clean, "object MathX {\n  def add(a: Int, b: Int): Int = a + b\n}\n"),
        // Go
        case(L::Go, Reject(LS), "func f() {\n\tdefer resp.Body.Close() //nolint:errcheck\n}\n"),
        case(L::Go, Reject(SKIP), "func TestAdd(t *testing.T) {\n\tt.Skip(\"later\")\n\tif add(1, 1) != 2 { t.Fatal(\"bad\") }\n}\n"),
        case(L::Go, Reject(ESC), "func g() {\n\terr := do()\n\t_ = err\n}\n"),
        case(L::Go, Reject(LS), "h := md5.New() // #nosec G401\n"),
        case(L::Go, Reject(SKIP), "//go:build ignore\n\npackage calc\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) { if add(1, 1) != 2 { t.Fatal() } }\n"),
        case(L::Go, Note(SKIP), "//go:build ignore\n\npackage main\n\nfunc main() { generate() }\n"),
        case(L::Go, Note(LS), "defer f.Close() //nolint:errcheck // read-only file, close error carries no information\n"),
        case(L::Go, Clean, "package calc\n\nfunc Add(a, b int) int { return a + b }\n"),
        case(L::Go, Clean, "package doc\n\nconst Tip = \"never write //nolint or t.Skip( in tests\"\n\nvar raw = `_ = err //nolint`\n"),
        // JavaScript
        case(L::JavaScript, Reject(LS), "/* eslint no-console: \"off\" */\nconsole.log(x);\n"),
        case(L::JavaScript, Reject(WEAK), "it.only('adds', () => {\n  expect(add(1, 1)).toBe(2);\n});\n"),
        case(L::JavaScript, Reject(SKIP), "test.todo('handles negatives');\n"),
        case(L::JavaScript, Reject(ESC), "try { save(); } catch (e) {}\n"),
        case(L::JavaScript, Reject(WEAK), "/* istanbul ignore next */\nit('adds', () => { expect(add(1, 1)).toBe(2); });\n"),
        case(L::JavaScript, Note(LS), "// prettier-ignore\nconst matrix = [1,0,\n                0,1];\n"),
        case(L::JavaScript, Note(LS), "// biome-ignore lint/suspicious/noExplicitAny: third-party callback signature is untyped\nfunction cb(x) { return x; }\n"),
        case(L::JavaScript, Clean, "export function add(a, b) {\n  return a + b;\n}\n"),
        case(L::JavaScript, Clean, "const tip = \"use /* istanbul ignore next */ or it.only( sparingly\";\nconst t = `// biome-ignore ${x}`;\n"),
        // TypeScript
        case(L::TypeScript, Reject(TYPE), "const user = JSON.parse(raw) as any;\n"),
        case(L::TypeScript, Reject(TYPE), "const n = (value as unknown as number) + 1;\n"),
        case(L::TypeScript, Reject(LS), "// deno-lint-ignore no-explicit-any\nfunction f(x: any) { return x; }\n"),
        case(L::TypeScript, Note(TYPE), "// reason: the SDK's typings omit this field; tracked upstream\nconst id = (resp as any).id;\n"),
        case(L::TypeScript, Clean, "export function add(a: number, b: number): number {\n  return a + b;\n}\n"),
        // Ruby
        case(L::Ruby, Reject(LS), "def call # rubocop:disable Metrics/AbcSize\n  work\nend\n"),
        case(L::Ruby, Reject(SKIP), "it 'adds' do\n  skip 'flaky on CI'\n  expect(add(1, 1)).to eq(2)\nend\n"),
        case(L::Ruby, Reject(SKIP), "xit 'adds' do\n  expect(add(1, 1)).to eq(2)\nend\n"),
        case(L::Ruby, Reject(STUB), "def area\n  raise NotImplementedError\nend\n"),
        case(L::Ruby, Reject(ESC), "value = Integer(input) rescue nil\n"),
        case(L::Ruby, Note(LS), "def call # rubocop:disable Metrics/AbcSize -- the parser table is generated and reviewed elsewhere\n  work\nend\n"),
        case(L::Ruby, Clean, "def add(a, b)\n  a + b\nend\n"),
        case(L::Ruby, Clean, "HELP = <<~TEXT\n  # rubocop:disable is not allowed here\n  skip 'x'\nTEXT\nputs HELP\n"),
        // PHP
        case(L::Php, Reject(LS), "<?php\n// @phpstan-ignore-next-line\n$x = $undefined + 1;\n"),
        case(L::Php, Reject(SKIP), "<?php\npublic function testAdd(): void {\n    $this->markTestSkipped('later');\n}\n"),
        case(L::Php, Reject(ESC), "<?php\n$data = @file_get_contents($path);\n"),
        case(L::Php, Reject(LS), "<?php\n$x = foo(); // phpcs:ignore Generic.PHP.ForbiddenFunctions\n"),
        case(L::Php, Note(LS), "<?php\n// @phpstan-ignore argument.type (the stub declares int but the extension returns a numeric string)\n$n = intdiv($a, 2);\n"),
        case(L::Php, Clean, "<?php\nfunction add(int $a, int $b): int {\n    return $a + $b;\n}\n"),
        case(L::Php, Clean, "<?php\n$help = <<<EOT\n// @phpstan-ignore-next-line is banned\nEOT;\n$mail = 'ops@example.com';\n"),
        // Swift
        case(L::Swift, Reject(LS), "// swiftlint:disable force_cast\nlet v = x as! Int\n"),
        case(L::Swift, Reject(SKIP), "func testLogin() throws {\n    throw XCTSkip(\"server down\")\n}\n"),
        case(L::Swift, Reject(STUB), "func parse() -> Int {\n    fatalError(\"not implemented\")\n}\n"),
        case(L::Swift, Reject(WEAK), "@Test func adds() {\n    withKnownIssue {\n        #expect(add(1, 1) == 3)\n    }\n}\n"),
        case(L::Swift, Note(LS), "// swiftlint:disable:next force_cast (reason: the cell type is registered two lines above)\nlet cell = dequeue() as! Cell\n"),
        case(L::Swift, Clean, "func add(_ a: Int, _ b: Int) -> Int { a + b }\n"),
        case(L::Swift, Clean, "required init?(coder: NSCoder) {\n    fatalError(\"init(coder:) has not been implemented\")\n}\n"),
        // Dart
        case(L::Dart, Reject(LS), "// ignore: avoid_print\nprint('x');\n"),
        case(L::Dart, Reject(SKIP), "test('adds', () {\n  expect(add(1, 1), 2);\n}, skip: 'flaky');\n"),
        case(L::Dart, Reject(STUB), "int parse(String s) => throw UnimplementedError();\n"),
        case(L::Dart, Reject(LS), "// ignore_for_file: unused_import\nimport 'dart:io';\n"),
        case(L::Dart, Note(LS), "// ignore: avoid_print (reason: this is the CLI's only output channel)\nprint('done');\n"),
        case(L::Dart, Clean, "int add(int a, int b) => a + b;\n"),
        case(L::Dart, Clean, "// Prefer fixing over adding an ignore: comment.\nint f() => 1;\n"),
        // Lua
        case(L::Lua, Reject(LS), "local x = 1 -- luacheck: ignore 211\n"),
        case(L::Lua, Reject(STUB), "local function parse(s)\n  error(\"not implemented\")\nend\n"),
        case(L::Lua, Reject(LS), "---@diagnostic disable-next-line: undefined-global\nprint(foo)\n"),
        case(L::Lua, Reject(SKIP), "pending(\"handles unicode\")\n"),
        case(L::Lua, Note(LS), "local x = 1 -- luacheck: ignore 211 (reason: kept for the C binding's field order)\n"),
        case(L::Lua, Clean, "local function add(a, b)\n  return a + b\nend\nreturn add\n"),
        case(L::Lua, Clean, "--[[ The luacheck: ignore option is documented in the README ]]\nlocal s = [[ -- luacheck: ignore ]]\nreturn s\n"),
        // Perl
        case(L::Perl, Reject(LS), "my $x = eval $code; ## no critic (ProhibitStringyEval)\n"),
        case(L::Perl, Reject(ESC), "no strict 'refs';\n*{\"main::$name\"} = sub { 1 };\n"),
        case(L::Perl, Reject(SKIP), "use Test::More;\nplan skip_all => 'needs network';\n"),
        case(L::Perl, Reject(STUB), "sub parse {\n    die \"not implemented\";\n}\n"),
        case(L::Perl, Note(LS), "## no critic (ProhibitStringyEval) -- reason: the plugin loader needs runtime module names\nmy $m = eval \"require $name\";\n"),
        case(L::Perl, Clean, "use strict;\nuse warnings;\nsub add { my ($a, $b) = @_; return $a + $b }\n1;\n"),
        case(L::Perl, Clean, "=pod\n\nNever use ## no critic in this module.\n\n=cut\n\n# no critic markers are banned here\nsub f { return scalar(@ARGV) + $#ARGV }\n"),
        // R
        case(L::R, Reject(LS), "x <- 1 # nolint\n"),
        case(L::R, Reject(SKIP), "test_that(\"adds\", {\n  skip(\"later\")\n  expect_equal(add(1, 1), 2)\n})\n"),
        case(L::R, Reject(SKIP), "test_that(\"adds\", {\n  skip_on_ci()\n  expect_equal(add(1, 1), 2)\n})\n"),
        case(L::R, Reject(ESC), "res <- suppressWarnings(as.numeric(x))\n"),
        case(L::R, Note(LS), "x <- 1 # nolint: object_name_linter. reason: matches the column name in the CSV\n"),
        case(L::R, Clean, "add <- function(a, b) {\n  a + b\n}\n"),
        // Julia
        case(L::Julia, Reject(SKIP), "@test_skip add(1, 1) == 2\n"),
        case(L::Julia, Reject(WEAK), "@test_broken add(1, 1) == 3\n"),
        case(L::Julia, Reject(STUB), "area(r) = error(\"not implemented\")\n"),
        case(L::Julia, Note(LS), "#! format: off\nconst M = [1 0\n           0 1]\n"),
        case(L::Julia, Clean, "add(a, b) = a + b\n@test add(1, 1) == 2\n"),
        // Haskell
        case(L::Haskell, Reject(LS), "{-# OPTIONS_GHC -Wno-unused-imports #-}\nmodule M where\n"),
        case(L::Haskell, Reject(STUB), "parse :: String -> Int\nparse = undefined\n"),
        case(L::Haskell, Reject(LS), "{- HLINT ignore \"Use map\" -}\nmodule M where\n"),
        case(L::Haskell, Reject(SKIP), "spec = it \"adds\" $ pending\n"),
        case(L::Haskell, Reject(STUB), "area r = error \"TODO\"\n"),
        case(L::Haskell, Note(LS), "{- HLINT ignore \"Use camelCase\" -} -- reason: names mirror the C API\nmodule M where\n"),
        case(L::Haskell, Clean, "add :: Int -> Int -> Int\nadd a b = a + b\n"),
        case(L::Haskell, Clean, "infixr 5 -->\n(-->) :: a -> b -> b\n_ --> b = b\n-- undefined behaviour is documented in the README\n"),
        // OCaml
        case(L::OCaml, Reject(LS), "let[@warning \"-32\"] unused x = x\n"),
        case(L::OCaml, Reject(STUB), "let parse s = failwith \"todo\"\n"),
        case(L::OCaml, Reject(TYPE), "let x : int = Obj.magic s\n"),
        case(L::OCaml, Note(LS), "(* reason: the binding is used only from C stubs *)\nlet[@warning \"-32\"] for_c x = x\n"),
        case(L::OCaml, Clean, "let add a b = a + b\n"),
        case(L::OCaml, Clean, "(* failwith \"todo\" is never acceptable *)\nlet f x = x\n"),
        // Elixir
        case(L::Elixir, Reject(LS), "# credo:disable-for-next-line\nIO.inspect(x)\n"),
        case(L::Elixir, Reject(SKIP), "@tag :skip\ntest \"adds\" do\n  assert add(1, 1) == 2\nend\n"),
        case(L::Elixir, Reject(STUB), "def parse(_s), do: raise \"not implemented\"\n"),
        case(L::Elixir, Reject(TYPE), "@dialyzer {:nowarn_function, run: 1}\ndef run(x), do: x\n"),
        case(L::Elixir, Note(LS), "# credo:disable-for-next-line Credo.Check.Warning.IoInspect -- reason: debug helper for the REPL only\nIO.inspect(x)\n"),
        case(L::Elixir, Clean, "defmodule MathX do\n  def add(a, b), do: a + b\nend\n"),
        case(L::Elixir, Clean, "x = ?\"\ndoc = \"\"\"\n@tag :skip is banned\n\"\"\"\n"),
        // Erlang
        case(L::Erlang, Reject(LS), "-compile(nowarn_unused_function).\n"),
        case(L::Erlang, Reject(TYPE), "-dialyzer({nowarn_function, run/1}).\n"),
        case(L::Erlang, Reject(STUB), "parse(_S) -> erlang:error(not_implemented).\n"),
        case(L::Erlang, Reject(SKIP), "init_per_testcase(_, _Config) -> {skip, \"needs db\"}.\n"),
        case(L::Erlang, Note(LS), "% reason: the helper is called through apply/3 from the plugin table\n-compile({nowarn_unused_function, [helper/1]}).\n"),
        case(L::Erlang, Clean, "-module(m).\n-export([add/2]).\nadd(A, B) -> A + B.\n"),
        case(L::Erlang, Clean, "q() -> $\".\n% not_implemented is a word we avoid\n"),
        // Clojure
        case(L::Clojure, Reject(SKIP), "#_(deftest adds\n  (is (= 2 (add 1 1))))\n"),
        case(L::Clojure, Reject(WEAK), "(deftest adds\n  (is true))\n"),
        case(L::Clojure, Reject(LS), "#_:clj-kondo/ignore\n(def x (undefined-fn))\n"),
        case(L::Clojure, Reject(SKIP), "(deftest ^:skip adds (is (= 2 (add 1 1))))\n"),
        case(L::Clojure, Note(LS), "; reason: the macro expands to a var kondo cannot see\n#_:clj-kondo/ignore\n(defmacro-generated foo)\n"),
        case(L::Clojure, Clean, "(defn add [a b] (+ a b))\n(deftest adds (is (= 2 (add 1 1))))\n"),
        case(L::Clojure, Clean, "(def tip \"#_(deftest x) hides a test\")\n(def q \\\")\n"),
        // Zig
        case(L::Zig, Reject(ESC), "pub fn main() void {\n    _ = std.fs.cwd().deleteFile(\"lock\");\n}\n"),
        case(L::Zig, Reject(STUB), "fn parse(s: []const u8) u32 {\n    @panic(\"TODO\");\n}\n"),
        case(L::Zig, Reject(SKIP), "test \"adds\" {\n    return error.SkipZigTest;\n}\n"),
        case(L::Zig, Reject(ESC), "const n = std.fmt.parseInt(u32, s, 10) catch unreachable;\n"),
        case(L::Zig, Reject(STUB), "fn parse(s: []const u8) u32 {\n    unreachable;\n}\n"),
        case(L::Zig, Note(ESC), "// reason: a missing lock file is the state we want\n_ = std.fs.cwd().deleteFile(\"lock\");\n"),
        case(L::Zig, Clean, "fn add(a: u32, b: u32) u32 {\n    return a + b;\n}\n"),
        case(L::Zig, Clean, "fn f(x: u32) void {\n    _ = x;\n    const s =\n        \\\\ @panic(\"TODO\")\n    ;\n    _ = s;\n}\n"),
        // Nim
        case(L::Nim, Reject(LS), "{.push warning[UnusedImport]: off.}\nimport os\n"),
        case(L::Nim, Reject(STUB), "proc parse(s: string): int = discard\n"),
        case(L::Nim, Reject(ESC), "{.push boundChecks: off.}\nproc sum(a: seq[int]): int =\n  for x in a: result += x\n"),
        case(L::Nim, Reject(STUB), "proc parse(s: string): int =\n  raise newException(ValueError, \"not implemented\")\n"),
        case(L::Nim, Note(LS), "{.push warning[XDeclaredButNotUsed]: off.} # reason: generated bindings declare every symbol\n"),
        case(L::Nim, Clean, "proc add(a, b: int): int = a + b\n"),
        case(L::Nim, Clean, "#[ {.push warning[X]: off.} is banned ]#\nproc f(): int = 1\n"),
        // Solidity
        case(L::Solidity, Reject(LS), "// solhint-disable-next-line avoid-low-level-calls\n(bool ok, ) = to.call{value: v}(\"\");\n"),
        case(L::Solidity, Reject(LS), "// slither-disable-next-line reentrancy-eth\nfunction withdraw() external { pay(); }\n"),
        case(L::Solidity, Reject(ESC), "function inc(uint x) pure returns (uint) {\n    unchecked { return x + 1; }\n}\n"),
        case(L::Solidity, Reject(STUB), "function quote() external returns (uint) {\n    revert(\"not implemented\");\n}\n"),
        case(L::Solidity, Note(ESC), "// reason: i < len is checked by the loop condition, so the increment cannot overflow\nunchecked { ++i; }\n"),
        case(L::Solidity, Clean, "pragma solidity ^0.8.20;\ncontract C {\n    function add(uint a, uint b) external pure returns (uint) { return a + b; }\n}\n"),
        // Shell
        case(L::Shell, Reject(LS), "# shellcheck disable=SC2086\nrm $files\n"),
        case(L::Shell, Reject(WEAK), "pytest -q || true\n"),
        case(L::Shell, Reject(ESC), "set +e\nmake test\n"),
        case(L::Shell, Reject(LS), "cargo clippy -- -A clippy::all\n"),
        case(L::Shell, Reject(CFG), "RUSTFLAGS=\"--cap-lints warn\" cargo build\n"),
        case(L::Shell, Reject(WEAK), "npx jest --passWithNoTests\n"),
        case(L::Shell, Reject(CFG), "git commit --no-verify -m wip\n"),
        case(L::Shell, Reject(SKIP), "./gradlew build -x test\n"),
        case(L::Shell, Note(LS), "# shellcheck disable=SC2086 # reason: word splitting of $FLAGS is intended\ncc $FLAGS main.c\n"),
        case(L::Shell, Clean, "#!/bin/sh\nset -eu\nmake test\nrm -f build.log || true\n"),
        case(L::Shell, Clean, "cat <<'EOF'\n# shellcheck disable=SC2086 is how you silence a check\nEOF\necho \"$# args\"\n"),
        // PowerShell
        case(L::PowerShell, Reject(LS), "[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '')]\nparam()\n"),
        case(L::PowerShell, Reject(ESC), "Remove-Item $path -ErrorAction SilentlyContinue\n"),
        case(L::PowerShell, Reject(ESC), "Set-StrictMode -Off\n"),
        case(L::PowerShell, Reject(SKIP), "It 'adds' -Skip {\n    Add 1 1 | Should -Be 2\n}\n"),
        case(L::PowerShell, Reject(STUB), "function Get-Thing {\n    throw 'not implemented'\n}\n"),
        case(L::PowerShell, Note(LS), "[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'Interactive installer must write to the host')]\nparam()\n"),
        case(L::PowerShell, Clean, "function Add-Numbers([int]$a, [int]$b) {\n    return $a + $b\n}\n"),
        case(L::PowerShell, Clean, "<#\n Never use -ErrorAction SilentlyContinue here.\n#>\nGet-Content log.txt | Select-Object -Skip 1\n"),
        // SQL
        case(L::Sql, Reject(LS), "SELECT * FROM t -- noqa\n"),
        case(L::Sql, Reject(ESC), "SET sql_mode = '';\n"),
        case(L::Sql, Reject(SKIP), "SELECT skip('needs extension pg_trgm');\n"),
        case(L::Sql, Reject(STUB), "CREATE FUNCTION f() RETURNS int AS $$\nBEGIN\n  RAISE EXCEPTION 'not implemented';\nEND;\n$$ LANGUAGE plpgsql;\n"),
        case(L::Sql, Note(LS), "SELECT * FROM t -- noqa: AM04 reason: the view is a passthrough by design\n"),
        case(L::Sql, Clean, "SELECT id, name FROM users WHERE active = 1;\n"),
        case(L::Sql, Clean, "INSERT INTO docs VALUES ('-- noqa is how sqlfluff skips a line');\n"),
        // Terraform
        case(L::Terraform, Reject(LS), "resource \"aws_s3_bucket\" \"b\" { #tfsec:ignore:aws-s3-enable-bucket-encryption\n  bucket = \"x\"\n}\n"),
        case(L::Terraform, Reject(LS), "resource \"aws_s3_bucket\" \"b\" {\n  #checkov:skip=CKV_AWS_18\n  bucket = \"x\"\n}\n"),
        case(L::Terraform, Reject(LS), "# tflint-ignore: terraform_unused_declarations\nvariable \"x\" {}\n"),
        case(L::Terraform, Note(LS), "#checkov:skip=CKV_AWS_20:The bucket hosts the public website by design\nresource \"aws_s3_bucket\" \"site\" {}\n"),
        case(L::Terraform, Clean, "resource \"aws_s3_bucket\" \"b\" {\n  bucket = \"logs\"\n}\n"),
        case(L::Terraform, Clean, "policy = <<EOF\n#tfsec:ignore:x\nEOF\n"),
        // Dockerfile
        case(L::Dockerfile, Reject(LS), "# hadolint ignore=DL3008\nRUN apt-get install -y curl\n"),
        case(L::Dockerfile, Reject(WEAK), "RUN pytest || true\n"),
        case(L::Dockerfile, Reject(LS), "# checkov:skip=CKV_DOCKER_2\nFROM alpine\n"),
        case(L::Dockerfile, Note(LS), "# hadolint ignore=DL3008 reason: the base image pins the mirror snapshot\nRUN apt-get install -y curl\n"),
        case(L::Dockerfile, Clean, "FROM rust:1.80\nRUN cargo test --locked\n"),
        case(L::Dockerfile, Clean, "RUN echo '# hadolint ignore=DL3008' > notes.txt\n"),
        // Makefile
        case(L::Makefile, Reject(WEAK), "test:\n\t-pytest -q\n"),
        case(L::Makefile, Reject(WEAK), ".IGNORE:\n"),
        case(L::Makefile, Reject(WEAK), "test:\n\t@true\n"),
        case(L::Makefile, Reject(WEAK), "check:\n\tcargo test || true\n"),
        case(L::Makefile, Reject(WEAK), "MAKEFLAGS += -i\n"),
        case(L::Makefile, Note(LS), "CFLAGS += -Wno-unused-parameter # reason: callbacks must match the vendor signature\n"),
        case(L::Makefile, Clean, "test:\n\tcargo test --locked\n\nclean:\n\t-rm -rf target\n"),
        // YAML
        case(L::Yaml, Reject(LS), "# yamllint disable-line rule:line-length\nkey: value\n"),
        case(L::Yaml, Reject(LS), "- name: install\n  shell: curl x | sh  # noqa: risky-shell-pipe\n"),
        case(L::Yaml, Reject(LS), "metadata:\n  annotations:\n    ignore-check.kube-linter.io/run-as-non-root: \"legacy\"\n"),
        case(L::Yaml, Note(LS), "# yamllint disable-line rule:line-length reason: the URL cannot be wrapped\nurl: https://example.com/very/long/path\n"),
        case(L::Yaml, Clean, "name: app\nreplicas: 2\n"),
        case(L::Yaml, Clean, "msg: 'use # yamllint disable sparingly'\n"),
        // CI
        case(L::CiConfig, Reject(WEAK), "jobs:\n  test:\n    runs-on: ubuntu-latest\n    continue-on-error: true\n    steps:\n      - run: cargo test\n"),
        case(L::CiConfig, Reject(WEAK), "test:\n  script: pytest\n  allow_failure: true\n"),
        case(L::CiConfig, Reject(SKIP), "  lint:\n    if: false\n    runs-on: ubuntu-latest\n"),
        case(L::CiConfig, Reject(WEAK), "      - run: cargo clippy || true\n"),
        case(L::CiConfig, Note(LS), "# yamllint disable-line rule:truthy reason: GitHub requires the bare on key\non: push\n"),
        case(L::CiConfig, Clean, "on: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: cargo test --locked\n"),
        // Cargo manifest
        case(L::CargoManifest, Reject(SKIP), "[lib]\ndoctest = false\n"),
        case(L::CargoManifest, Reject(SKIP), "[[test]]\nname = \"integration\"\nharness = false\n"),
        case(L::CargoManifest, Reject(CFG), "[alias]\nclippy = \"check\"\n"),
        case(L::CargoManifest, Reject(WEAK), "[profile.test]\ndebug-assertions = false\n"),
        case(L::CargoManifest, Reject(SKIP), "[package]\nname = \"x\"\nautotests = false\n"),
        case(L::CargoManifest, Reject(CFG), "[lints.clippy]\nunwrap_used = \"allow\"\n"),
        case(L::CargoManifest, Note(SKIP), "[lib]\ndoctest = false # reason: the crate has no public doc examples, only the C header\n"),
        case(L::CargoManifest, Clean, "[package]\nname = \"calc\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lints.clippy]\nunwrap_used = \"deny\"\n"),
        case(L::CargoManifest, Clean, "[[bench]]\nname = \"speed\"\nharness = false\n"),
        // clippy.toml
        case(L::ClippyConfig, Reject(CFG), "too-many-arguments-threshold = 999\n"),
        case(L::ClippyConfig, Reject(CFG), "cognitive-complexity-threshold = 120\n"),
        case(L::ClippyConfig, Note(CFG), "too-many-lines-threshold = 400 # reason: generated parser tables live in one function\n"),
        case(L::ClippyConfig, Note(CFG), "allow-unwrap-in-tests = true\n"),
        case(L::ClippyConfig, Clean, "too-many-arguments-threshold = 8\nmsrv = \"1.80\"\n"),
        // Python config
        case(L::PythonConfig, Reject(LS), "[tool.ruff.lint]\nignore = [\"E501\", \"F401\"]\n"),
        case(L::PythonConfig, Reject(CFG), "[mypy]\nignore_errors = True\n"),
        case(L::PythonConfig, Reject(TYPE), "[tool.pyright]\ntypeCheckingMode = \"off\"\n"),
        case(L::PythonConfig, Reject(LS), "[pytest]\naddopts = -q --disable-warnings\n"),
        case(L::PythonConfig, Reject(CFG), "[coverage:report]\nfail_under = 0\n"),
        case(L::PythonConfig, Reject(WEAK), "[tox]\nignore_outcome = true\n"),
        case(L::PythonConfig, Note(LS), "[flake8]\n# reason: black owns line length, so E501 would only duplicate it\nextend-ignore = E501\n"),
        case(L::PythonConfig, Clean, "[tool.ruff]\nline-length = 100\nselect = [\"E\", \"F\"]\n"),
        // JS lint config
        case(L::JsLintConfig, Reject(CFG), "{\n  \"rules\": {\n    \"no-console\": \"off\"\n  }\n}\n"),
        case(L::JsLintConfig, Reject(CFG), "rules:\n  eqeqeq: 0\n"),
        case(L::JsLintConfig, Reject(CFG), "{ \"linter\": { \"enabled\": false } }\n"),
        case(L::JsLintConfig, Reject(CFG), "module.exports = { ignorePatterns: [\"**/*\"] };\n"),
        case(L::JsLintConfig, Note(CFG), "{\n  \"rules\": {\n    // reason: the CLI prints to stdout by design\n    \"no-console\": \"off\"\n  }\n}\n"),
        case(L::JsLintConfig, Clean, "{\n  \"extends\": \"eslint:recommended\",\n  \"rules\": { \"eqeqeq\": \"error\" }\n}\n"),
        // tsconfig
        case(L::TsConfig, Reject(TYPE), "{ \"compilerOptions\": { \"strict\": false } }\n"),
        case(L::TsConfig, Reject(TYPE), "{\n  \"compilerOptions\": {\n    \"noImplicitAny\": false\n  }\n}\n"),
        case(L::TsConfig, Reject(TYPE), "{ \"compilerOptions\": { \"suppressImplicitAnyIndexErrors\": true } }\n"),
        case(L::TsConfig, Note(TYPE), "{ \"compilerOptions\": { \"skipLibCheck\": true } }\n"),
        case(L::TsConfig, Clean, "{ \"compilerOptions\": { \"strict\": true, \"target\": \"es2022\" } }\n"),
        // package.json / test-runner config
        case(L::NodePackage, Reject(WEAK), "{\n  \"scripts\": {\n    \"test\": \"exit 0\"\n  }\n}\n"),
        case(L::NodePackage, Reject(WEAK), "{ \"scripts\": { \"test\": \"jest --passWithNoTests\" } }\n"),
        case(L::NodePackage, Reject(WEAK), "{ \"scripts\": { \"lint\": \"eslint . || true\" } }\n"),
        case(L::NodePackage, Note(SKIP), "module.exports = {\n  // reason: the e2e suite runs in its own job with a browser\n  testPathIgnorePatterns: ['/e2e/'],\n};\n"),
        case(L::NodePackage, Clean, "{\n  \"scripts\": {\n    \"test\": \"jest --ci\",\n    \"lint\": \"eslint .\"\n  }\n}\n"),
        // golangci
        case(L::GolangciConfig, Reject(CFG), "run:\n  issues-exit-code: 0\n"),
        case(L::GolangciConfig, Reject(LS), "linters:\n  disable:\n    - errcheck\n"),
        case(L::GolangciConfig, Reject(CFG), "run:\n  tests: false\n"),
        case(L::GolangciConfig, Note(CFG), "linters:\n  disable-all: true\n  enable:\n    - govet\n"),
        case(L::GolangciConfig, Clean, "linters:\n  enable:\n    - govet\n    - staticcheck\n"),
        // MSBuild
        case(L::MsBuild, Reject(LS), "<PropertyGroup>\n  <NoWarn>CS1591;CS0618</NoWarn>\n</PropertyGroup>\n"),
        case(L::MsBuild, Reject(CFG), "<PropertyGroup>\n  <WarningLevel>0</WarningLevel>\n</PropertyGroup>\n"),
        case(L::MsBuild, Reject(CFG), "<RunAnalyzers>false</RunAnalyzers>\n"),
        case(L::MsBuild, Note(LS), "<!-- reason: CS1591 fires on generated protobuf types we do not document -->\n<NoWarn>CS1591</NoWarn>\n"),
        case(L::MsBuild, Clean, "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>\n  </PropertyGroup>\n</Project>\n"),
        // .editorconfig
        case(L::EditorConfig, Reject(LS), "[*.cs]\ndotnet_diagnostic.CA2007.severity = none\n"),
        case(L::EditorConfig, Reject(CFG), "[*.cs]\ndotnet_analyzer_diagnostic.severity = none\n"),
        case(L::EditorConfig, Reject(CFG), "[gen/**.cs]\ngenerated_code = true\n"),
        case(L::EditorConfig, Note(LS), "[*.cs]\n# reason: ConfigureAwait is irrelevant in an app with no sync context\ndotnet_diagnostic.CA2007.severity = none\n"),
        case(L::EditorConfig, Clean, "root = true\n[*]\nindent_style = space\n"),
        // Gradle
        case(L::Gradle, Reject(WEAK), "checkstyle {\n    ignoreFailures = true\n}\n"),
        case(L::Gradle, Reject(SKIP), "test {\n    enabled = false\n}\n"),
        case(L::Gradle, Reject(SKIP), "tasks.test {\n    filter { excludeTestsMatching(\"*Slow*\") }\n}\n"),
        case(L::Gradle, Note(SKIP), "tasks.test {\n    // reason: the device suite runs only on the emulator farm\n    filter { excludeTestsMatching(\"*DeviceTest\") }\n}\n"),
        case(L::Gradle, Clean, "plugins { id(\"java\") }\ntasks.test { useJUnitPlatform() }\n"),
        // Maven
        case(L::Maven, Reject(SKIP), "<properties>\n  <skipTests>true</skipTests>\n</properties>\n"),
        case(L::Maven, Reject(WEAK), "<testFailureIgnore>true</testFailureIgnore>\n"),
        case(L::Maven, Reject(CFG), "<properties><checkstyle.skip>true</checkstyle.skip></properties>\n"),
        case(L::Maven, Reject(LS), "<suppressions>\n  <suppress checks=\".*\" files=\".*\"/>\n</suppressions>\n"),
        case(L::Maven, Note(CFG), "<!-- reason: the shade plugin runs only in the release profile -->\n<skip>true</skip>\n"),
        case(L::Maven, Clean, "<project>\n  <modelVersion>4.0.0</modelVersion>\n</project>\n"),
        // RuboCop config
        case(L::RubocopConfig, Reject(CFG), "Metrics/MethodLength:\n  Enabled: false\n"),
        case(L::RubocopConfig, Reject(CFG), "AllCops:\n  DisabledByDefault: true\n"),
        case(L::RubocopConfig, Reject(CFG), "Metrics/AbcSize:\n  Max: 500\n"),
        case(L::RubocopConfig, Note(CFG), "inherit_from: .rubocop_todo.yml\n"),
        case(L::RubocopConfig, Clean, "AllCops:\n  NewCops: enable\n  TargetRubyVersion: 3.2\n"),
        // PHP lint config
        case(L::PhpLintConfig, Reject(CFG), "parameters:\n  level: 0\n"),
        case(L::PhpLintConfig, Reject(CFG), "parameters:\n  ignoreErrors:\n    - '#Call to an undefined method#'\n"),
        case(L::PhpLintConfig, Reject(CFG), "<psalm errorLevel=\"8\">\n</psalm>\n"),
        case(L::PhpLintConfig, Note(CFG), "parameters:\n  # reason: the legacy module is excluded until the PHP 8 port lands\n  ignoreErrors:\n    - '#legacy#'\n"),
        case(L::PhpLintConfig, Clean, "parameters:\n  level: 8\n  paths:\n    - src\n"),
        // SwiftLint config
        case(L::SwiftlintConfig, Reject(CFG), "disabled_rules:\n  - force_cast\n"),
        case(L::SwiftlintConfig, Reject(CFG), "allow_zero_lintable_files: true\n"),
        case(L::SwiftlintConfig, Note(CFG), "# reason: generated code is linted by its own generator's checks\nexcluded:\n  - Generated\n"),
        case(L::SwiftlintConfig, Clean, "opt_in_rules:\n  - empty_count\n"),
        // Dart analysis_options
        case(L::DartAnalysis, Reject(LS), "analyzer:\n  errors:\n    unused_import: ignore\n"),
        case(L::DartAnalysis, Reject(CFG), "linter:\n  rules:\n    avoid_print: false\n"),
        case(L::DartAnalysis, Note(LS), "analyzer:\n  errors:\n    # reason: the generated l10n files import everything\n    unused_import: ignore\n"),
        case(L::DartAnalysis, Clean, "include: package:lints/recommended.yaml\n"),
        // CMake
        case(L::CMake, Reject(LS), "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -w\")\n"),
        case(L::CMake, Reject(SKIP), "set_tests_properties(slow PROPERTIES DISABLED TRUE)\n"),
        case(L::CMake, Reject(CFG), "add_compile_options(-Wno-error)\n"),
        case(L::CMake, Note(LS), "# reason: the vendored zlib triggers this warning and is not ours to fix\nadd_compile_options(-Wno-deprecated-non-prototype)\n"),
        case(L::CMake, Clean, "cmake_minimum_required(VERSION 3.20)\nproject(calc C)\nadd_executable(calc main.c)\n"),
        // .clang-tidy
        case(L::ClangTidyConfig, Reject(CFG), "Checks: '-*'\n"),
        case(L::ClangTidyConfig, Reject(LS), "Checks: 'clang-analyzer-*,-clang-analyzer-security.insecureAPI.strcpy'\n"),
        case(L::ClangTidyConfig, Note(LS), "# reason: the codebase targets C89, so the modernize checks cannot apply\nChecks: 'bugprone-*,-modernize-*'\n"),
        case(L::ClangTidyConfig, Clean, "Checks: 'clang-analyzer-*,bugprone-*'\nWarningsAsErrors: '*'\n"),
    ]
}

/// Per-language regression numbers.
#[derive(Debug, Clone, Serialize)]
pub struct LangRate {
    pub lang: Lang,
    pub positives: usize,
    pub detected: usize,
    pub notes: usize,
    pub notes_matched: usize,
    pub negatives: usize,
    pub false_positives: usize,
    pub recall: f64,
    pub fp_rate: f64,
}

/// Per-kind regression recall.
#[derive(Debug, Clone, Serialize)]
pub struct KindRate {
    pub kind: PolyKind,
    pub positives: usize,
    pub detected: usize,
    pub recall: f64,
}

/// Scores over [`regression_cases`]: a regression corpus, not a detection
/// rate (see the module docs).
#[derive(Debug, Clone, Serialize)]
pub struct Rates {
    pub cases: usize,
    pub positives: usize,
    pub detected: usize,
    pub recall: f64,
    pub notes: usize,
    pub notes_matched: usize,
    pub note_accuracy: f64,
    pub negatives: usize,
    pub false_positives: usize,
    pub fp_rate: f64,
    pub per_lang: Vec<LangRate>,
    pub per_kind: Vec<KindRate>,
    /// `lang#index: expected ... got ...` for every case that missed.
    pub misses: Vec<String>,
    /// Distinct rules in the catalog.
    pub rules_total: usize,
    /// Distinct rules that fired on at least one case: how much of the
    /// catalog the corpus exercises at all.
    pub rules_fired: usize,
}

/// `num / den` as a rate; an empty denominator is a rate of 1 for recall-like
/// numbers and 0 for error-like ones, chosen by `empty`.
fn ratio(num: usize, den: usize, empty: f64) -> f64 {
    let (Ok(n), Ok(d)) = (u32::try_from(num), u32::try_from(den)) else {
        return empty;
    };
    if d == 0 {
        empty
    } else {
        f64::from(n) / f64::from(d)
    }
}

/// Whether `hits` satisfy `expect`.
pub fn satisfies(expect: Expect, hits: &[Hit]) -> bool {
    match expect {
        Expect::Reject(k) => hits
            .iter()
            .any(|h| h.kind == k && h.severity > Severity::Note),
        Expect::Note(k) => {
            hits.iter()
                .any(|h| h.kind == k && h.severity == Severity::Note)
                && hits.iter().all(|h| h.severity == Severity::Note)
        }
        Expect::Clean => hits.is_empty(),
    }
}

/// Score [`regression_cases`].
pub fn regression_rates() -> Rates {
    let cases = regression_cases();
    let mut fired = std::collections::BTreeSet::new();
    let mut misses = Vec::new();
    let mut per_lang: Vec<LangRate> = Lang::ALL
        .iter()
        .map(|&lang| LangRate {
            lang,
            positives: 0,
            detected: 0,
            notes: 0,
            notes_matched: 0,
            negatives: 0,
            false_positives: 0,
            recall: 1.0,
            fp_rate: 0.0,
        })
        .collect();
    let mut per_kind: Vec<KindRate> = PolyKind::ALL
        .iter()
        .map(|&kind| KindRate {
            kind,
            positives: 0,
            detected: 0,
            recall: 1.0,
        })
        .collect();
    for (n, c) in cases.iter().enumerate() {
        let hits = scan(c.text, c.lang);
        fired.extend(hits.iter().map(|h| h.rule));
        let ok = satisfies(c.expect, &hits);
        if !ok {
            let got: Vec<String> = hits
                .iter()
                .map(|h| format!("{}@{}:{}", h.rule, h.line.unwrap_or(0), h.severity.as_str()))
                .collect();
            misses.push(format!(
                "{}#{n}: expected {:?}, got [{}]",
                c.lang.as_str(),
                c.expect,
                got.join(", ")
            ));
        }
        if let Some(lr) = per_lang.iter_mut().find(|r| r.lang == c.lang) {
            match c.expect {
                Expect::Reject(_) => {
                    lr.positives += 1;
                    lr.detected += usize::from(ok);
                }
                Expect::Note(_) => {
                    lr.notes += 1;
                    lr.notes_matched += usize::from(ok);
                }
                Expect::Clean => {
                    lr.negatives += 1;
                    lr.false_positives += usize::from(!ok);
                }
            }
        }
        if let Expect::Reject(kind) = c.expect {
            if let Some(kr) = per_kind.iter_mut().find(|r| r.kind == kind) {
                kr.positives += 1;
                kr.detected += usize::from(ok);
            }
        }
    }
    for lr in &mut per_lang {
        lr.recall = ratio(lr.detected, lr.positives, 1.0);
        lr.fp_rate = ratio(lr.false_positives, lr.negatives, 0.0);
    }
    for kr in &mut per_kind {
        kr.recall = ratio(kr.detected, kr.positives, 1.0);
    }
    let sum = |f: fn(&LangRate) -> usize| per_lang.iter().map(f).sum::<usize>();
    let (positives, detected) = (sum(|r| r.positives), sum(|r| r.detected));
    let (notes, notes_matched) = (sum(|r| r.notes), sum(|r| r.notes_matched));
    let (negatives, false_positives) = (sum(|r| r.negatives), sum(|r| r.false_positives));
    let rules_total = Lang::ALL
        .iter()
        .flat_map(|&l| applicable_rules(l).map(|r| r.id))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    Rates {
        cases: cases.len(),
        positives,
        detected,
        recall: ratio(detected, positives, 1.0),
        notes,
        notes_matched,
        note_accuracy: ratio(notes_matched, notes, 1.0),
        negatives,
        false_positives,
        fp_rate: ratio(false_positives, negatives, 0.0),
        per_lang,
        per_kind,
        misses,
        rules_total,
        rules_fired: fired.len(),
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
mod tests {
    use super::*;

    fn rules_hit(text: &str, lang: Lang) -> Vec<&'static str> {
        scan(text, lang).iter().map(|h| h.rule).collect()
    }

    #[test]
    fn test_regression_corpus_is_fully_classified() {
        let rates = regression_rates();
        eprintln!(
            "regression corpus: {} cases, {} positives, {} notes, {} negatives; {} of {} rules fired",
            rates.cases,
            rates.positives,
            rates.notes,
            rates.negatives,
            rates.rules_fired,
            rates.rules_total
        );
        assert!(
            rates.misses.is_empty(),
            "regression misses:\n{}",
            rates.misses.join("\n")
        );
        assert_eq!(rates.recall, 1.0);
        assert_eq!(rates.note_accuracy, 1.0);
        assert_eq!(rates.fp_rate, 0.0);
        assert!(rates.rules_fired > 0 && rates.rules_fired <= rates.rules_total);
        for kr in &rates.per_kind {
            assert!(
                kr.positives >= 2,
                "kind {} has {} positives",
                kr.kind.as_str(),
                kr.positives
            );
        }
    }

    #[test]
    fn test_every_language_has_positives_a_note_and_a_negative() {
        let cases = regression_cases();
        for &lang in Lang::ALL {
            let of = |f: fn(Expect) -> bool| {
                cases
                    .iter()
                    .filter(|c| c.lang == lang && f(c.expect))
                    .count()
            };
            assert!(
                of(|e| matches!(e, Expect::Reject(_))) >= 2,
                "{} needs two positives",
                lang.as_str()
            );
            assert!(
                of(|e| matches!(e, Expect::Note(_))) >= 1,
                "{} needs a note case",
                lang.as_str()
            );
            assert!(
                of(|e| e == Expect::Clean) >= 1,
                "{} needs a clean case",
                lang.as_str()
            );
        }
        assert!(Lang::ALL.len() >= 30);
    }

    #[test]
    fn test_catalog_ids_are_unique_and_prefixed() {
        let mut seen = std::collections::BTreeMap::new();
        for &lang in Lang::ALL {
            assert!(!rules(lang).is_empty(), "{} has no rules", lang.as_str());
            for rule in applicable_rules(lang) {
                assert!(rule.id.contains('.'), "{}", rule.id);
                assert!(!rule.because.is_empty());
                let first = seen.entry(rule.id).or_insert(rule.because);
                assert_eq!(*first, rule.because, "id {} reused for two rules", rule.id);
            }
        }
    }

    #[test]
    fn test_names_round_trip_and_aliases_resolve() {
        for &lang in Lang::ALL {
            assert_eq!(Lang::from_name(lang.as_str()), Some(lang));
        }
        assert_eq!(Lang::from_name("C++"), Some(Lang::Cpp));
        assert_eq!(Lang::from_name("golang"), Some(Lang::Go));
        assert_eq!(Lang::from_name("bash"), Some(Lang::Shell));
        assert_eq!(Lang::from_name("github-actions"), Some(Lang::CiConfig));
        assert_eq!(Lang::from_name("brainfuck"), None);
    }

    #[test]
    fn test_paths_extensions_and_shebangs() {
        assert_eq!(Lang::from_extension(".tsx"), Some(Lang::TypeScript));
        assert_eq!(Lang::from_extension("toml"), None);
        assert_eq!(
            Lang::from_path("repo/.github/workflows/ci.yml"),
            Some(Lang::CiConfig)
        );
        assert_eq!(Lang::from_path("deploy/k8s.yaml"), Some(Lang::Yaml));
        assert_eq!(Lang::from_path("Dockerfile.dev"), Some(Lang::Dockerfile));
        assert_eq!(Lang::from_path("src\\Makefile"), Some(Lang::Makefile));
        assert_eq!(Lang::from_path("pyproject.toml"), Some(Lang::PythonConfig));
        assert_eq!(
            Lang::from_path("web/.eslintrc.cjs"),
            Some(Lang::JsLintConfig)
        );
        assert_eq!(Lang::from_path("tsconfig.build.json"), Some(Lang::TsConfig));
        assert_eq!(
            Lang::from_path("crate/Cargo.toml"),
            Some(Lang::CargoManifest)
        );
        assert_eq!(
            Lang::from_path(".cargo/config.toml"),
            Some(Lang::CargoManifest)
        );
        assert_eq!(Lang::from_path("clippy.toml"), Some(Lang::ClippyConfig));
        assert_eq!(Lang::from_path("app/build.gradle.kts"), Some(Lang::Gradle));
        assert_eq!(Lang::from_path("App.csproj"), Some(Lang::MsBuild));
        assert_eq!(Lang::from_path("CMakeLists.txt"), Some(Lang::CMake));
        assert_eq!(Lang::from_path("main.rs"), Some(Lang::Rust));
        assert_eq!(Lang::from_path("README"), None);
        assert_eq!(Lang::from_path(".gitignore"), None);
        assert_eq!(
            Lang::from_shebang("#!/usr/bin/env python3.12"),
            Some(Lang::Python)
        );
        assert_eq!(
            Lang::from_shebang("#!/usr/bin/env -S node --no-warnings"),
            Some(Lang::JavaScript)
        );
        assert_eq!(Lang::from_shebang("#!/bin/bash -e"), Some(Lang::Shell));
        assert_eq!(
            Lang::from_shebang("#!/usr/bin/make -f"),
            Some(Lang::Makefile)
        );
        assert_eq!(Lang::from_shebang("// not a shebang"), None);
    }

    #[test]
    fn test_a_pyi_stub_file_declares_rather_than_stubs() {
        let stub = "def area(r: float) -> float:\n    ...\n";
        // The same text is an empty function as an answer, a declaration in a
        // `.pyi` stub file.
        assert!(scan(stub, Lang::Python)
            .iter()
            .any(|h| h.rule == "py.ellipsis-body"));
        assert!(scan_path(stub, Lang::Python, Some("pkg/shapes.pyi")).is_empty());
        assert_eq!(
            scan_path(stub, Lang::Python, Some("pkg/shapes.py")),
            scan(stub, Lang::Python)
        );
        // Only the stub rule is lifted: a suppression in a stub file is one.
        let ignored = "def area(r: float) -> float: ...  # type: ignore\n# mypy: ignore-errors\n";
        assert!(!scan_path(ignored, Lang::Python, Some("pkg/shapes.pyi")).is_empty());
    }

    #[test]
    fn test_snippet_family_agrees_with_langdata() {
        use crate::langdata::snippet_language;
        use crate::student::CodeLanguage;
        assert_eq!(
            Lang::Go.snippet_family(),
            snippet_language(CodeLanguage::Go)
        );
        assert_eq!(
            Lang::Java.snippet_family(),
            snippet_language(CodeLanguage::Java)
        );
        assert_eq!(
            Lang::CSharp.snippet_family(),
            snippet_language(CodeLanguage::CSharp)
        );
        assert_eq!(
            Lang::Php.snippet_family(),
            snippet_language(CodeLanguage::Php)
        );
        assert_eq!(Lang::Rust.snippet_family(), SnippetLanguage::Rust);
        assert_eq!(Lang::Kotlin.snippet_family(), SnippetLanguage::Other);
        assert_eq!(Lang::Yaml.snippet_family(), SnippetLanguage::Other);
    }

    #[test]
    fn test_views_are_byte_aligned_and_utf8() {
        let src = "let s = \"héllo // not a comment\"; // réal comment\nlet c = 'é';\n";
        let doc = lex(src, Lang::Rust);
        for line in &doc.lines {
            assert_eq!(line.code.len(), line.raw.len());
            assert_eq!(line.text.len(), line.raw.len());
        }
        let first = &doc.lines[0];
        assert!(!first.code.contains("not a comment"));
        assert!(first.text.contains("not a comment"));
        assert_eq!(first.comments.len(), 1);
        assert!(first.comment_text().contains("réal comment"));
    }

    #[test]
    fn test_rust_raw_strings_and_lifetimes_lex_as_data_and_code() {
        let src = "fn f<'a>(x: &'a str) -> &'a str { x }\nconst R: &str = r##\"\"# #[cfg(any())] \"##;\nconst C: char = '\\'';\n";
        let doc = lex(src, Lang::Rust);
        assert!(doc.lines[0].code.contains("fn f<'a>"));
        assert!(!doc.lines[1].code.contains("cfg"));
        assert!(rules_hit(src, Lang::Rust).is_empty());
    }

    #[test]
    fn test_multi_line_attribute_beyond_a_four_line_window() {
        let src = "#[\n\n\n\n\ncfg(\nany()\n)\n]\n#[test]\nfn hidden() { assert!(false); }\n";
        assert!(rules_hit(src, Lang::Rust).contains(&"rust.cfg-false"));
    }

    #[test]
    fn test_directive_position_separates_directives_from_prose() {
        assert!(rules_hit("x = 1  # nosec\n", Lang::Python).contains(&"py.nosec"));
        assert!(rules_hit("x = 1  # a  # nosec\n", Lang::Python).contains(&"py.nosec"));
        assert!(rules_hit("# never use nosec here\nx = 1\n", Lang::Python).is_empty());
        assert!(rules_hit("s = '# nosec'\n", Lang::Python).is_empty());
        // A block comment's directive may start on its second line.
        let js = "/*\n  biome-ignore lint: reason\n*/\nf();\n";
        assert!(rules_hit(js, Lang::JavaScript).contains(&"js.biome-ignore"));
    }

    #[test]
    fn test_heredocs_and_pod_are_data() {
        let sh = "cat <<EOF\n# shellcheck disable=SC2086\nEOF\nls\n";
        assert!(rules_hit(sh, Lang::Shell).is_empty());
        let sh_indented =
            "\tcat <<-END\n\t# shellcheck disable=SC1\n\tEND\n# shellcheck disable=SC2\n";
        assert_eq!(
            rules_hit(sh_indented, Lang::Shell),
            vec!["sh.shellcheck-disable"]
        );
        let pl = "=head1 Notes\n\n## no critic\n\n=cut\nmy $x = 1;\n";
        assert!(rules_hit(pl, Lang::Perl).is_empty());
    }

    #[test]
    fn test_justification_downgrades_only_waivable_rules() {
        let bare = scan("defer f.Close() //nolint:errcheck\n", Lang::Go);
        assert_eq!(worst(&bare), Some(Severity::Suppression));
        let native = scan(
            "defer f.Close() //nolint:errcheck // close error is irrelevant here\n",
            Lang::Go,
        );
        assert_eq!(worst(&native), Some(Severity::Note));
        // A placeholder is not prose.
        let filler = scan("defer f.Close() //nolint:errcheck // ok\n", Lang::Go);
        assert_eq!(worst(&filler), Some(Severity::Suppression));
        // A skipped test stays skipped whatever the comment says.
        let skip = scan(
            "func TestX(t *testing.T) {\n\tt.Skip() // reason: flaky on the CI runner\n}\n",
            Lang::Go,
        );
        assert_eq!(worst(&skip), Some(Severity::Suppression));
    }

    #[test]
    fn test_escalation_marks_coverage_exclusion_on_tests() {
        let helper = scan(
            "/* c8 ignore next */\nfunction log() {}\n",
            Lang::JavaScript,
        );
        assert_eq!(worst(&helper), Some(Severity::Note));
        let on_test = scan(
            "/* c8 ignore next */\ntest('x', () => { expect(1).toBe(1); });\n",
            Lang::JavaScript,
        );
        assert_eq!(worst(&on_test), Some(Severity::Evasion));
    }

    #[test]
    fn test_or_true_needs_a_gate_command() {
        assert_eq!(worst(&scan("rm -f lock || true\n", Lang::Shell)), None);
        assert_eq!(
            worst(&scan("cargo test || true\n", Lang::Shell)),
            Some(Severity::Fraud)
        );
        assert_eq!(
            worst(&scan("      - run: npm test || true\n", Lang::CiConfig)),
            Some(Severity::Fraud)
        );
    }

    #[test]
    fn test_hits_carry_kind_tool_and_location() {
        let overlapping = scan("[lib]\ntest = false\n", Lang::CargoManifest);
        assert_eq!(overlapping.len(), 1);
        assert!(rule_by_id(overlapping[0].rule)
            .and_then(|r| r.overlap)
            .is_some());
        assert!(rule_by_id("no.such-rule").is_none());
        let hits = scan("x\n// solhint-disable-next-line\ny\n", Lang::Solidity);
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!(h.kind, PolyKind::LintSuppression);
        assert_eq!(h.lang, Lang::Solidity);
        assert_eq!(h.line, Some(2));
        assert_eq!(h.tool, Some("solhint"));
        assert_eq!(h.evidence, "// solhint-disable-next-line");
    }

    #[test]
    fn test_empty_and_odd_input_never_panics() {
        for &lang in Lang::ALL {
            assert!(scan("", lang).is_empty());
            drop(scan("\"unterminated\n'\n/*\n#[\n<<EOF\n@\"\n[[\n", lang));
            drop(scan("é\u{202e}'\\", lang));
        }
    }
}
