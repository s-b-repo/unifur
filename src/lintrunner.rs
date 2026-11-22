//! Per-language external linters ("clippy for every language"), run safely
//! on untrusted snippets when they are installed.
//!
//! [`registry`] lists real linters and type checkers per [`Lang`] with the
//! flags that make them read one snippet in isolation: no project config, no
//! plugins, no build scripts. [`run`] executes one; [`lint_all`] runs every
//! installed linter for a language. [`crate::codequality::external`] runs a
//! single configured command; this module generalizes it per language and
//! keeps its rule that output which cannot be read is never scored clean.
//!
//! # Safety model
//!
//! The source is untrusted model output, so the runner treats every tool
//! invocation as hostile input to a trusted program:
//!
//! - The source is written only into a fresh private directory (mode `0700`,
//!   unique name, created with `create`, not `create_all`, so an existing
//!   path is never reused) and removed on every exit path by a drop guard.
//! - The child runs with that directory as its working directory, a cleared
//!   environment (`PATH` with relative entries dropped; `HOME`, `TMPDIR` and
//!   the XDG directories pointed into the private directory, so no user
//!   configuration is read; only toolchain *location* variables such as
//!   `RUSTUP_HOME` and `PYTHONUSERBASE` passed through), stdin/stdout/stderr
//!   piped, its own process group, and a hard timeout after which the child
//!   and its group are killed.
//! - stdin is fed from a separate thread and stdout/stderr are drained by
//!   reader threads with a size cap, so neither a tool that stops reading nor
//!   one that floods its output can deadlock or exhaust the caller (the
//!   deadlock [`crate::codequality::external`] documents).
//! - The program is resolved to an absolute path through `PATH` before it
//!   runs, and arguments are passed as an argument vector: no source text is
//!   ever handed to a shell. The one interpreter entry (PSScriptAnalyzer via
//!   `pwsh -Command`) passes a constant command naming only the generated
//!   file path.
//! - A tool that is not installed is [`LintStatus::Unavailable`], one that
//!   would execute the analysed code is [`LintStatus::Refused`] unless the
//!   caller sets [`RunOptions::allow_executing`], and a crash, timeout or
//!   unreadable output is never a clean pass: [`LintOutcome::clean`] is true
//!   only for a completed run with zero diagnostics.
//!
//! # What this does not claim
//!
//! This is isolation by configuration, not a sandbox. The child runs as the
//! calling user and can read what that user can read: a C `#include` of an
//! absolute path or a Rust `include_str!` makes the compiler read that file,
//! and a diagnostic can echo part of it. Tools that walk *parent*
//! directories for configuration (clang-tidy, ktlint, Dart, SwiftLint,
//! sqlfluff) can still find a file planted above the temp root; pass
//! [`RunOptions::temp_root`] under a directory you control to close that. A
//! timeout kills the process group, but a descendant that leaves the group
//! can outlive it. Flags are written against current releases; each entry's
//! `notes` says which ones are version-sensitive. Use a container or a
//! seccomp sandbox when the threat model needs more.

use crate::polyglot::Lang;
use anyhow::{bail, Context, Result};
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How the source reaches the tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// Piped on stdin. `ext` names the synthetic file `snippet.<ext>`
    /// substituted for `<name>` in the arguments; `filename_arg` records the
    /// flag that carries it, when the tool needs one to pick its rules.
    Stdin {
        filename_arg: Option<&'static str>,
        ext: &'static str,
    },
    /// Written to `snippet.<ext>` in the private directory; `<file>` in the
    /// arguments is replaced by that relative path.
    TempFile { ext: &'static str },
}

impl Input {
    fn file_name(self) -> String {
        let ext = match self {
            Input::Stdin { ext, .. } | Input::TempFile { ext } => ext,
        };
        if ext.len() > 1 && ext.chars().next().is_some_and(char::is_uppercase) {
            // Whole file names (`Dockerfile`) are used as given.
            ext.to_string()
        } else {
            format!("snippet.{ext}")
        }
    }
}

/// How diagnostics are counted from the tool's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// One JSON value per line.
    JsonLines,
    /// One JSON document (array or object).
    Json,
    /// A SARIF 2.1 log: `runs[].results[]`.
    Sarif,
    /// Exit status only: non-zero is one finding.
    ExitCodeOnly,
    /// Every non-empty line is one finding (`gofmt -l`).
    LineCount,
    /// Lines that carry a source location (`file:12:`, `file(12,5)`,
    /// `line 12`) are findings; other lines are chatter.
    Located,
}

/// Which stream carries the diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
    Both,
}

/// One registered linter.
#[derive(Debug, Clone, Copy)]
pub struct Linter {
    pub lang: Lang,
    /// Display name (`"clippy"`, `"ruff"`).
    pub tool: &'static str,
    /// Executable looked up on `PATH`.
    pub program: &'static str,
    /// Arguments; `<file>` and `<name>` are substituted (see [`Input`]).
    pub args: &'static [&'static str],
    pub input: Input,
    pub output: Output,
    pub stream: Stream,
    /// Whether running the tool runs the analysed code (compile-time
    /// evaluation, build files, `BEGIN` blocks). Never run unless the caller
    /// opts in.
    pub executes_code: bool,
    /// Support files written next to the snippet: (file name, contents).
    pub support: &'static [(&'static str, &'static str)],
    /// Extra environment for the child, after the minimal base.
    pub env: &'static [(&'static str, &'static str)],
    /// Caveats: version-sensitive flags, network use, what the run cannot see.
    pub notes: &'static str,
}

const GO_ENV: &[(&str, &str)] = &[
    ("GOTOOLCHAIN", "local"),
    ("GOPROXY", "off"),
    ("GOFLAGS", "-mod=mod -modcacherw"),
    ("CGO_ENABLED", "0"),
    ("GOTELEMETRY", "off"),
];
const GO_MOD: &[(&str, &str)] = &[("go.mod", "module snippet\n\ngo 1.18\n")];
const DOTNET_ENV: &[(&str, &str)] = &[
    ("DOTNET_CLI_TELEMETRY_OPTOUT", "1"),
    ("DOTNET_NOLOGO", "1"),
    ("DOTNET_SKIP_FIRST_TIME_EXPERIENCE", "1"),
];
const CSPROJ: &[(&str, &str)] = &[(
    "snippet.csproj",
    "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n    <OutputType>Library</OutputType>\n    <Nullable>enable</Nullable>\n    <EnableNETAnalyzers>true</EnableNETAnalyzers>\n    <AnalysisLevel>latest</AnalysisLevel>\n  </PropertyGroup>\n</Project>\n",
)];
const SOLHINT_CONFIG: &[(&str, &str)] =
    &[(".solhint.json", "{\"extends\": \"solhint:recommended\"}\n")];

const fn linter(
    lang: Lang,
    tool: &'static str,
    program: &'static str,
    args: &'static [&'static str],
    input: Input,
    (output, stream): (Output, Stream),
    notes: &'static str,
) -> Linter {
    Linter {
        lang,
        tool,
        program,
        args,
        input,
        output,
        stream,
        executes_code: false,
        support: &[],
        env: &[],
        notes,
    }
}

impl Linter {
    const fn executes(mut self) -> Self {
        self.executes_code = true;
        self
    }
    const fn support(mut self, files: &'static [(&'static str, &'static str)]) -> Self {
        self.support = files;
        self
    }
    const fn env(mut self, vars: &'static [(&'static str, &'static str)]) -> Self {
        self.env = vars;
        self
    }
}

const fn stdin(ext: &'static str) -> Input {
    Input::Stdin {
        filename_arg: None,
        ext,
    }
}

const fn stdin_named(flag: &'static str, ext: &'static str) -> Input {
    Input::Stdin {
        filename_arg: Some(flag),
        ext,
    }
}

const fn file(ext: &'static str) -> Input {
    Input::TempFile { ext }
}

use Output::{ExitCodeOnly, Json, JsonLines, LineCount, Located, Sarif};
use Stream::{Both, Stderr, Stdout};

const ESLINT_ARGS: &[&str] = &[
    "--stdin",
    "--stdin-filename",
    "<name>",
    "--format",
    "json",
    "--no-config-lookup",
    "--rule",
    "no-unused-vars: error",
    "--rule",
    "no-unreachable: error",
    "--rule",
    "no-dupe-keys: error",
    "--rule",
    "no-empty: error",
    "--rule",
    "no-debugger: error",
    "--rule",
    "no-self-compare: error",
    "--rule",
    "no-constant-condition: error",
];

static REGISTRY: &[Linter] = &[
    // Rust. `clippy-driver` on one file, never `cargo clippy`: cargo would
    // run the build scripts and proc macros of whatever crate it finds.
    linter(Lang::Rust, "clippy", "clippy-driver", &["--edition", "2021", "--crate-type", "lib", "--emit=metadata", "--error-format=json", "-W", "clippy::all", "<file>"], file("rs"), (JsonLines, Stderr),
        "single-file lint; no dependencies resolve, so `use` of an external crate is itself a diagnostic. `include_str!` can read local files at compile time"),
    linter(Lang::Rust, "rustc", "rustc", &["--edition", "2021", "--crate-type", "lib", "--emit=metadata", "--error-format=json", "<file>"], file("rs"), (JsonLines, Stderr),
        "compiler warnings only; `include_str!` can read local files at compile time"),
    // Python.
    linter(Lang::Python, "ruff", "ruff", &["check", "--isolated", "--output-format", "json", "--stdin-filename", "<name>", "-"], stdin_named("--stdin-filename", "py"), (Json, Stdout),
        "`--isolated` ignores every configuration file; default rule set (E4, E7, E9, F)"),
    linter(Lang::Python, "pyright", "pyright", &["--outputjson", "<file>"], file("py"), (Json, Stdout),
        "the PyPI `pyright` wrapper downloads Node and pyright on first use (network); the npm package does not"),
    linter(Lang::Python, "mypy", "mypy", &["--strict", "--no-error-summary", "--output", "json", "<file>"], file("py"), (JsonLines, Stdout),
        "`--output json` needs mypy 1.11 or later; older versions fail, which is reported, not passed"),
    linter(Lang::Python, "bandit", "bandit", &["-q", "-f", "json", "-"], stdin("py"), (Json, Stdout),
        "reads `-` as stdin in current releases; `.bandit` configs are not read from the private directory"),
    // JavaScript / TypeScript.
    linter(Lang::JavaScript, "eslint", "eslint", ESLINT_ARGS, stdin_named("--stdin-filename", "js"), (Json, Stdout),
        "flat-config ESLint (8.57+/9.x): `--no-config-lookup` plus inline `--rule` flags, so no project config or plugin (which would execute JS) is loaded; ESLint 8 in eslintrc mode rejects the flag, reported as a failure"),
    linter(Lang::JavaScript, "biome", "biome", &["lint", "--reporter=json", "<file>"], file("js"), (Json, Stdout),
        "`--reporter=json` is marked experimental by Biome; output shape `{diagnostics: [...]}`"),
    linter(Lang::JavaScript, "node --check", "node", &["--check", "<file>"], file("js"), (ExitCodeOnly, Stderr),
        "syntax check only: `--check` parses without running; one finding per failing file"),
    linter(Lang::TypeScript, "tsc", "tsc", &["--noEmit", "--strict", "--pretty", "false", "<file>"], file("ts"), (Located, Stdout),
        "a file argument makes tsc ignore tsconfig.json; DOM and ES libs only, so Node globals are diagnostics"),
    linter(Lang::TypeScript, "biome", "biome", &["lint", "--reporter=json", "<file>"], file("ts"), (Json, Stdout),
        "`--reporter=json` is marked experimental by Biome"),
    // Shell.
    linter(Lang::Shell, "shellcheck", "shellcheck", &["--norc", "-f", "json", "-"], stdin("sh"), (Json, Stdout),
        "`--norc` ignores `.shellcheckrc`; without a shebang SC2148 is reported"),
    // Go.
    linter(Lang::Go, "go vet", "go", &["vet", "."], file("go"), (Located, Stderr),
        "temporary module (`go.mod`, go 1.18); `GOTOOLCHAIN=local` and `GOPROXY=off` forbid downloads; cgo off").support(GO_MOD).env(GO_ENV),
    linter(Lang::Go, "staticcheck", "staticcheck", &["-f", "json", "."], file("go"), (JsonLines, Stdout),
        "temporary module; no downloads").support(GO_MOD).env(GO_ENV),
    linter(Lang::Go, "gofmt", "gofmt", &["-l", "<file>"], file("go"), (LineCount, Stdout),
        "formatting only: lists the file when it is not gofmt-clean"),
    // C / C++ / Objective-C.
    linter(Lang::C, "clang-tidy", "clang-tidy", &["--quiet", "--config={Checks: 'clang-analyzer-*,bugprone-*'}", "<file>", "--", "-std=c11"], file("c"), (Located, Stdout),
        "`--config` overrides any `.clang-tidy` file; `--` means no compilation database"),
    linter(Lang::C, "cppcheck", "cppcheck", &["--quiet", "--enable=warning,style,performance,portability", "--template={file}:{line}:{column}: {severity}: {message} [{id}]", "<file>"], file("c"), (Located, Stderr),
        "`--inline-suppr` is deliberately not passed, so in-source suppressions are not honored"),
    linter(Lang::Cpp, "clang-tidy", "clang-tidy", &["--quiet", "--config={Checks: 'clang-analyzer-*,bugprone-*,modernize-*'}", "<file>", "--", "-std=c++17"], file("cpp"), (Located, Stdout),
        "`--config` overrides any `.clang-tidy` file; `--` means no compilation database"),
    linter(Lang::Cpp, "cppcheck", "cppcheck", &["--quiet", "--language=c++", "--enable=warning,style,performance,portability", "--template={file}:{line}:{column}: {severity}: {message} [{id}]", "<file>"], file("cpp"), (Located, Stderr),
        "`--inline-suppr` is deliberately not passed"),
    linter(Lang::ObjectiveC, "clang-tidy", "clang-tidy", &["--quiet", "--config={Checks: 'clang-analyzer-*,bugprone-*'}", "<file>", "--", "-x", "objective-c"], file("m"), (Located, Stdout),
        "Foundation headers are absent off macOS, so imports are diagnostics there"),
    // C#.
    linter(Lang::CSharp, "roslyn", "dotnet", &["build", "-nologo", "-clp:NoSummary", "--no-restore"], file("cs"), (Located, Stdout),
        "MSBuild evaluates project files (code execution) and restore downloads packages; refused unless executing is allowed, and `--no-restore` needs a warm package cache").support(CSPROJ).env(DOTNET_ENV).executes(),
    // Java / Kotlin / Scala.
    linter(Lang::Java, "checkstyle", "checkstyle", &["-c", "/google_checks.xml", "-f", "sarif", "<file>"], file("java"), (Sarif, Stdout),
        "the `checkstyle` launcher name varies by distribution (`java -jar checkstyle.jar`); SARIF output needs 8.42+"),
    linter(Lang::Java, "pmd", "pmd", &["check", "--no-cache", "--no-progress", "-R", "rulesets/java/quickstart.xml", "-f", "sarif", "-d", "<file>"], file("java"), (Sarif, Stdout),
        "PMD 7 command line; PMD 6 used `run.sh pmd` and is reported as a failure"),
    linter(Lang::Kotlin, "ktlint", "ktlint", &["--stdin", "--reporter=json"], stdin("kt"), (Json, Stdout),
        "ktlint reads `.editorconfig` from the working directory and its parents"),
    linter(Lang::Scala, "scalac", "scalac", &["-deprecation", "-feature", "-Xlint", "-d", ".", "<file>"], file("scala"), (Located, Both),
        "compiles to the private directory; `-Xlint` spelling differs between Scala 2 and 3"),
    // Ruby.
    linter(Lang::Ruby, "rubocop", "rubocop", &["--stdin", "<name>", "--format", "json", "--force-default-config", "--cache", "false"], stdin_named("--stdin", "rb"), (Json, Stdout),
        "`--force-default-config` ignores `.rubocop.yml`, and with it every `require:` plugin"),
    linter(Lang::Ruby, "ruby -wc", "ruby", &["-wc", "<file>"], file("rb"), (Located, Stderr),
        "`-c` compiles without running; `BEGIN` blocks are not executed under `-c`"),
    // PHP.
    linter(Lang::Php, "phpstan", "phpstan", &["analyse", "--no-progress", "--no-interaction", "--error-format=json", "--level=5", "--memory-limit=512M", "<file>"], file("php"), (Json, Stdout),
        "static reflection only; no autoload file is passed, so nothing is included"),
    linter(Lang::Php, "php -l", "php", &["-n", "-l", "<file>"], file("php"), (ExitCodeOnly, Both),
        "syntax check only; `-n` skips php.ini so no `auto_prepend_file` applies"),
    // Swift, Dart.
    linter(Lang::Swift, "swiftlint", "swiftlint", &["lint", "--use-stdin", "--reporter", "json", "--no-cache", "--quiet"], stdin("swift"), (Json, Stdout),
        "reads `.swiftlint.yml` from the working directory (the private one)"),
    linter(Lang::Dart, "dart analyze", "dart", &["analyze", "--format=json", "<file>"], file("dart"), (Json, Stdout),
        "`analysis_options.yaml` in a parent directory can enable analyzer plugins; set `temp_root`"),
    // Haskell, Lua, Perl, R, Julia, Zig, Nim, Clojure, Elixir.
    linter(Lang::Haskell, "hlint", "hlint", &["--json", "-"], stdin("hs"), (Json, Stdout), "reads `.hlint.yaml` from the working directory"),
    linter(Lang::Lua, "luacheck", "luacheck", &["--no-config", "--formatter", "plain", "--codes", "-"], stdin("lua"), (Located, Stdout),
        "`--no-config` matters: `.luacheckrc` is a Lua script"),
    linter(Lang::Perl, "perlcritic", "perlcritic", &["--noprofile", "--quiet", "--verbose", "%f:%l:%c: %m [%p]\\n", "<file>"], file("pl"), (Located, Stdout),
        "`--noprofile` ignores `.perlcriticrc`; PPI parses without running"),
    linter(Lang::Perl, "perl -c", "perl", &["-c", "-w", "<file>"], file("pl"), (Located, Stderr),
        "`perl -c` runs `BEGIN` blocks and `use` imports: it executes code").executes(),
    linter(Lang::R, "lintr", "Rscript", &["--vanilla", "-e", "print(lintr::lint('snippet.R'))"], file("R"), (Located, Stdout),
        "`--vanilla` skips `.Rprofile`; lintr parses without evaluating"),
    linter(Lang::Zig, "zig ast-check", "zig", &["ast-check", "<file>"], file("zig"), (Located, Stderr),
        "parse and AST checks only; comptime code is not evaluated"),
    linter(Lang::Nim, "nim check", "nim", &["check", "--hints:off", "<file>"], file("nim"), (Located, Both),
        "compile-time macros, `static:` blocks and `staticExec` run during `nim check`").executes(),
    linter(Lang::Clojure, "clj-kondo", "clj-kondo", &["--lint", "-", "--config-dir", ".", "--config", "{:output {:format :json}}"], stdin("clj"), (Json, Stdout),
        "`--config-dir .` keeps project hooks (SCI code) out"),
    linter(Lang::Elixir, "credo", "mix", &["credo", "--format", "json", "--strict", "<file>"], file("ex"), (Json, Stdout),
        "needs a Mix project with credo as a dependency; `mix` evaluates `mix.exs`").executes(),
    // Solidity, PowerShell, SQL, Terraform, Dockerfile, YAML, CI.
    linter(Lang::Solidity, "solhint", "solhint", &["--formatter", "json", "<file>"], file("sol"), (Json, Stdout),
        "config is the written `.solhint.json` (`solhint:recommended`); JSON formatter shape varies by major version").support(SOLHINT_CONFIG),
    linter(Lang::PowerShell, "PSScriptAnalyzer", "pwsh", &["-NoProfile", "-NonInteractive", "-NoLogo", "-Command", "$ErrorActionPreference = 'Stop'; Import-Module PSScriptAnalyzer; ConvertTo-Json -Depth 3 -InputObject @(Invoke-ScriptAnalyzer -Path ./snippet.ps1 | Select-Object RuleName,Severity,Line,Message)"], file("ps1"), (Json, Stdout),
        "constant command naming only the generated file. `$ErrorActionPreference = 'Stop'` and the explicit `Import-Module` matter: without them a missing module printed `[]` and exited 0, a false clean pass"),
    linter(Lang::Sql, "sqlfluff", "sqlfluff", &["lint", "-", "--format", "json", "--dialect", "ansi", "--nocolor"], stdin("sql"), (Json, Stdout),
        "`--dialect ansi`; sqlfluff also looks for config in parent directories"),
    linter(Lang::Terraform, "tflint", "tflint", &["--format=json", "--no-color"], file("tf"), (Json, Stdout),
        "lints the working directory; plugins are not installed in the private HOME, so only built-in rules run"),
    linter(Lang::Dockerfile, "hadolint", "hadolint", &["--no-color", "-f", "json", "-"], stdin("Dockerfile"), (Json, Stdout), "reads `.hadolint.yaml` from the private directory only"),
    linter(Lang::Yaml, "yamllint", "yamllint", &["-f", "parsable", "-d", "default", "-"], stdin("yaml"), (Located, Stdout), "`-d default` ignores `.yamllint` files"),
    linter(Lang::CiConfig, "actionlint", "actionlint", &["-format", "{{json .}}", "-"], stdin("yml"), (Json, Stdout),
        "GitHub Actions only; runs shellcheck on `run:` blocks when installed"),
    linter(Lang::CiConfig, "yamllint", "yamllint", &["-f", "parsable", "-d", "default", "-"], stdin("yml"), (Located, Stdout), "`-d default` ignores `.yamllint` files"),
];

/// Every registered linter.
pub fn registry() -> &'static [Linter] {
    REGISTRY
}

/// The registered linters for one language.
pub fn for_lang(lang: Lang) -> impl Iterator<Item = &'static Linter> {
    REGISTRY.iter().filter(move |l| l.lang == lang)
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// How a run ended. Only [`LintStatus::Completed`] can be clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintStatus {
    /// The tool ran to completion and its output was read.
    Completed,
    /// The tool was killed at the timeout.
    TimedOut,
    /// The program is not on `PATH`.
    Unavailable,
    /// The tool executes the analysed code and the caller did not opt in.
    Refused,
    /// The tool could not be run, or its output could not be read.
    Failed,
}

/// The result of one linter on one snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintOutcome {
    pub tool: &'static str,
    pub status: LintStatus,
    pub diagnostics: usize,
    /// Diagnostic messages (or, on failure, the reason), truncated.
    pub messages: Vec<String>,
    pub exit: Option<i32>,
    pub timed_out: bool,
    /// Output beyond [`RunOptions::max_output`] was discarded.
    pub truncated: bool,
}

impl LintOutcome {
    /// A completed run with no diagnostics: the only outcome that is a pass.
    pub fn clean(&self) -> bool {
        self.status == LintStatus::Completed && self.diagnostics == 0
    }

    fn not_run(tool: &'static str, status: LintStatus, why: String) -> Self {
        Self {
            tool,
            status,
            diagnostics: 0,
            messages: vec![why],
            exit: None,
            timed_out: false,
            truncated: false,
        }
    }
}

/// Knobs for [`run_with`] and [`lint_all`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOptions {
    /// Hard limit per tool; the process group is killed at the deadline.
    pub timeout: Duration,
    /// Run tools whose [`Linter::executes_code`] is set.
    pub allow_executing: bool,
    /// Bytes kept per output stream.
    pub max_output: usize,
    /// Messages kept per outcome.
    pub max_messages: usize,
    /// Where the private directory is created; defaults to the system
    /// temporary directory.
    pub temp_root: Option<PathBuf>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            allow_executing: false,
            max_output: 1 << 20,
            max_messages: 50,
            temp_root: None,
        }
    }
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    p.metadata().is_ok_and(|m| m.is_file())
}

/// The absolute path `program` resolves to on `PATH`, skipping relative
/// entries. Nothing is executed.
pub fn resolve(program: &str) -> Option<PathBuf> {
    if program.contains('/') {
        let p = PathBuf::from(program);
        return (p.is_absolute() && is_executable(&p)).then_some(p);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(program))
        .find(|p| is_executable(p))
}

/// Whether the linter's program is installed: a `PATH` lookup, nothing run.
pub fn available(linter: &Linter) -> bool {
    resolve(linter.program).is_some()
}

/// A private directory removed on drop, whatever path leaves the scope.
struct PrivateDir {
    path: PathBuf,
}

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

impl PrivateDir {
    fn create(root: Option<&Path>) -> Result<Self> {
        let root = root.map_or_else(std::env::temp_dir, Path::to_path_buf);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        for _ in 0..16 {
            let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("unifur-lint-{}-{nanos}-{n}", std::process::id()));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(e).with_context(|| {
                        format!("create private lint directory {}", path.display())
                    })
                }
            }
        }
        bail!(
            "could not create a fresh private directory under {}",
            root.display()
        )
    }

    fn subdir(&self, name: &str) -> Result<()> {
        let path = self.path.join(name);
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&path)
            .with_context(|| format!("create {}", path.display()))
    }

    fn write(&self, name: &str, content: &str) -> Result<()> {
        let plain =
            !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0']);
        if !plain {
            bail!("refusing support file name {name:?}: not a plain file name");
        }
        let path = self.path.join(name);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut f = options
            .open(&path)
            .with_context(|| format!("create {}", path.display()))?;
        f.write_all(content.as_bytes())
            .with_context(|| format!("write {}", path.display()))
    }
}

impl Drop for PrivateDir {
    fn drop(&mut self) {
        // Best effort: a drop cannot report, and a leftover directory under
        // the temp root holds only the snippet the caller already has.
        drop(std::fs::remove_dir_all(&self.path));
    }
}

/// The child's `TMPDIR`, inside the private directory.
const TMP_SUBDIR: &str = ".tmp";

/// Toolchain *location* variables: where an installed tool finds its own
/// runtime. None of them is a linter configuration path.
const LOCATION_VARS: &[&str] = &[
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "CARGO_HOME",
    "PYTHONUSERBASE",
    "GEM_HOME",
    "GEM_PATH",
    "GOROOT",
    "JAVA_HOME",
    "DOTNET_ROOT",
    "PSModulePath",
];

fn child_env(dir: &Path, extra: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
    let mut env: Vec<(OsString, OsString)> = Vec::new();
    if let Some(path) = std::env::var_os("PATH") {
        let absolute: Vec<PathBuf> = std::env::split_paths(&path)
            .filter(|p| p.is_absolute())
            .collect();
        if let Ok(joined) = std::env::join_paths(absolute) {
            env.push(("PATH".into(), joined));
        }
    }
    for var in LOCATION_VARS {
        if let Some(v) = std::env::var_os(var) {
            env.push(((*var).into(), v));
        }
    }
    // A tool installed per user is found through the real home even though
    // the child's HOME is the private directory.
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for (var, sub) in [
            ("RUSTUP_HOME", ".rustup"),
            ("CARGO_HOME", ".cargo"),
            ("PYTHONUSERBASE", ".local"),
        ] {
            let derived = home.join(sub);
            if std::env::var_os(var).is_none() && derived.is_dir() {
                env.push((var.into(), derived.into_os_string()));
            }
        }
    }
    let under = |sub: &str| dir.join(sub).into_os_string();
    env.push(("HOME".into(), dir.as_os_str().to_os_string()));
    // A subdirectory, not the working directory itself: Go ignores a
    // `go.mod` that sits in the temp root.
    env.push(("TMPDIR".into(), under(TMP_SUBDIR)));
    env.push(("XDG_CONFIG_HOME".into(), under(".config")));
    env.push(("XDG_CACHE_HOME".into(), under(".cache")));
    env.push(("XDG_DATA_HOME".into(), under(".local/share")));
    env.push(("XDG_STATE_HOME".into(), under(".local/state")));
    env.push(("LC_ALL".into(), "C.UTF-8".into()));
    env.push(("NO_COLOR".into(), "1".into()));
    env.push(("TERM".into(), "dumb".into()));
    for (k, v) in extra {
        env.push(((*k).into(), (*v).into()));
    }
    env
}

/// Drain a pipe on its own thread, keeping at most `cap` bytes.
fn spawn_reader<R: Read + Send + 'static>(
    mut pipe: R,
    cap: usize,
) -> mpsc::Receiver<(Vec<u8>, bool)> {
    let (tx, rx) = mpsc::channel();
    // Detached on purpose: the receiver waits with a timeout, so a reader
    // held open by a stray grandchild can never hang the caller.
    drop(std::thread::spawn(move || {
        let mut kept = Vec::new();
        let limit = u64::try_from(cap).unwrap_or(u64::MAX);
        let mut truncated = false;
        if (&mut pipe).take(limit).read_to_end(&mut kept).is_ok() {
            let mut sink = std::io::sink();
            truncated = std::io::copy(&mut pipe, &mut sink).is_ok_and(|n| n > 0);
        }
        // The receiver may already be gone after a timeout; nothing to tell.
        drop(tx.send((kept, truncated)));
    }));
    rx
}

/// Kill the child and, best effort, its process group.
fn kill_tree(child: &mut Child, notes: &mut Vec<String>) {
    #[cfg(unix)]
    if let Some(kill) = resolve("kill") {
        let group = format!("-{}", child.id());
        let status = Command::new(kill)
            .args(["-KILL", "--", group.as_str()])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if let Err(e) = status {
            notes.push(format!("could not signal the process group: {e}"));
        }
    }
    if let Err(e) = child.kill() {
        notes.push(format!("could not kill the tool: {e}"));
    }
    if let Err(e) = child.wait() {
        notes.push(format!("could not reap the tool: {e}"));
    }
}

/// Run one linter with default options and the given timeout.
pub fn run(linter: &Linter, source: &str, timeout: Duration) -> Result<LintOutcome> {
    run_with(
        linter,
        source,
        &RunOptions {
            timeout,
            ..RunOptions::default()
        },
    )
}

/// Run one linter. `Err` when the tool is unavailable, refused, cannot be
/// run, or its output cannot be read; a timeout is an `Ok` outcome with
/// [`LintStatus::TimedOut`], which is not [`LintOutcome::clean`].
pub fn run_with(linter: &Linter, source: &str, opts: &RunOptions) -> Result<LintOutcome> {
    let Some(program) = resolve(linter.program) else {
        bail!(
            "{} is not installed (no executable `{}` on PATH); an unavailable linter is not a clean pass",
            linter.tool,
            linter.program
        );
    };
    if linter.executes_code && !opts.allow_executing {
        bail!(
            "refusing to run {}: it executes the analysed code ({})",
            linter.tool,
            linter.notes
        );
    }
    let dir = PrivateDir::create(opts.temp_root.as_deref())?;
    dir.subdir(TMP_SUBDIR)?;
    for (name, content) in linter.support {
        dir.write(name, content)?;
    }
    let name = linter.input.file_name();
    let piped = matches!(linter.input, Input::Stdin { .. });
    if !piped {
        dir.write(&name, source)?;
    }
    let args: Vec<String> = linter
        .args
        .iter()
        .map(|a| a.replace("<file>", &name).replace("<name>", &name))
        .collect();
    let mut cmd = Command::new(&program);
    cmd.args(&args)
        .current_dir(&dir.path)
        .env_clear()
        .envs(child_env(&dir.path, linter.env))
        .stdin(if piped { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {} ({})", linter.tool, program.display()))?;
    let writer = child.stdin.take().map(|mut stdin| {
        let bytes = source.as_bytes().to_vec();
        let (tx, rx) = mpsc::channel();
        // Fed from its own thread: writing first and reading after deadlocks
        // once both pipe buffers fill.
        drop(std::thread::spawn(move || {
            let result = stdin.write_all(&bytes).map_err(|e| e.to_string());
            drop(stdin);
            drop(tx.send(result));
        }));
        rx
    });
    let out_rx = child
        .stdout
        .take()
        .map(|p| spawn_reader(p, opts.max_output));
    let err_rx = child
        .stderr
        .take()
        .map(|p| spawn_reader(p, opts.max_output));

    let start = Instant::now();
    let mut notes = Vec::new();
    let status: Option<ExitStatus> = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if start.elapsed() >= opts.timeout => {
                kill_tree(&mut child, &mut notes);
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                kill_tree(&mut child, &mut notes);
                return Err(e).with_context(|| format!("wait for {}", linter.tool));
            }
        }
    };
    let grace = Duration::from_secs(2);
    let collect = |rx: Option<mpsc::Receiver<(Vec<u8>, bool)>>| {
        rx.and_then(|rx| rx.recv_timeout(grace).ok())
            .unwrap_or_default()
    };
    let (stdout, out_cut) = collect(out_rx);
    let (stderr, err_cut) = collect(err_rx);
    let truncated = out_cut || err_cut;
    let Some(status) = status else {
        let mut messages = notes;
        messages.insert(0, format!("killed after {:?}", opts.timeout));
        return Ok(LintOutcome {
            tool: linter.tool,
            status: LintStatus::TimedOut,
            diagnostics: 0,
            messages,
            exit: None,
            timed_out: true,
            truncated,
        });
    };
    if let Some(rx) = writer {
        match rx.recv_timeout(grace) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => bail!(
                "{} stopped reading the source ({e}); a partial read is not a clean pass",
                linter.tool
            ),
            Err(_) => bail!("{}: the stdin writer did not finish", linter.tool),
        }
    }
    let selected = match linter.stream {
        Stream::Stdout => String::from_utf8_lossy(&stdout).into_owned(),
        Stream::Stderr => String::from_utf8_lossy(&stderr).into_owned(),
        Stream::Both => format!(
            "{}\n{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ),
    };
    let exit = status.code();
    let (diagnostics, mut messages) = parse_output(linter, &selected, status.success())?;
    messages.truncate(opts.max_messages);
    messages.extend(notes);
    Ok(LintOutcome {
        tool: linter.tool,
        status: LintStatus::Completed,
        diagnostics,
        messages,
        exit,
        timed_out: false,
        truncated,
    })
}

/// Run every registered linter for `lang`. Tools that are not installed,
/// that are refused, or that fail appear as non-clean outcomes, so an empty
/// or all-unavailable result can never be mistaken for a pass.
pub fn lint_all(source: &str, lang: Lang, opts: &RunOptions) -> Vec<LintOutcome> {
    lint_with(for_lang(lang), source, opts)
}

fn lint_with<'a>(
    linters: impl Iterator<Item = &'a Linter>,
    source: &str,
    opts: &RunOptions,
) -> Vec<LintOutcome> {
    linters
        .map(|l| {
            if !available(l) {
                return LintOutcome::not_run(
                    l.tool,
                    LintStatus::Unavailable,
                    format!("`{}` is not on PATH", l.program),
                );
            }
            if l.executes_code && !opts.allow_executing {
                return LintOutcome::not_run(
                    l.tool,
                    LintStatus::Refused,
                    format!("executes the analysed code: {}", l.notes),
                );
            }
            run_with(l, source, opts).unwrap_or_else(|e| {
                LintOutcome::not_run(l.tool, LintStatus::Failed, format!("{e:#}"))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

fn truncate_message(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() > 200 {
        line.chars().take(197).collect::<String>() + "..."
    } else {
        line.to_string()
    }
}

/// Count diagnostics and collect their messages. A non-zero exit with no
/// diagnostics, or output that cannot be read in the declared format, is an
/// error: a crashed or misconfigured tool is not a clean file.
fn parse_output(linter: &Linter, out: &str, success: bool) -> Result<(usize, Vec<String>)> {
    let (count, messages) = match linter.output {
        Output::ExitCodeOnly => {
            let messages: Vec<String> = out
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(truncate_message)
                .collect();
            return Ok((
                usize::from(!success),
                if success { Vec::new() } else { messages },
            ));
        }
        Output::LineCount => {
            let lines: Vec<String> = out
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(truncate_message)
                .collect();
            (lines.len(), lines)
        }
        Output::Located => {
            let lines: Vec<String> = out
                .lines()
                .filter(|l| is_located(l))
                .map(truncate_message)
                .collect();
            (lines.len(), lines)
        }
        Output::Json | Output::Sarif | Output::JsonLines => {
            let values = json_values(out, linter.output == Output::JsonLines);
            if values.is_empty() {
                if linter.output == Output::JsonLines && out.trim().is_empty() && success {
                    return Ok((0, Vec::new()));
                }
                bail!(
                    "{} produced no readable {} output; refusing to score it as clean: {}",
                    linter.tool,
                    if linter.output == Output::Sarif {
                        "SARIF"
                    } else {
                        "JSON"
                    },
                    truncate_message(out)
                );
            }
            let mut found = Vec::new();
            for v in &values {
                collect_diagnostics(v, &mut found);
            }
            let messages = found.iter().copied().filter_map(message_of).collect();
            (found.len(), messages)
        }
    };
    if !success && count == 0 {
        bail!(
            "{} exited unsuccessfully and reported no diagnostics; refusing to score it as clean: {}",
            linter.tool,
            truncate_message(out)
        );
    }
    Ok((count, messages))
}

/// A line that names a source location: `name:12:`, `name(12,5)`, `line 12`.
fn is_located(line: &str) -> bool {
    let b = line.as_bytes();
    let digits_then = |mut k: usize, end: u8| {
        let start = k;
        while b.get(k).is_some_and(u8::is_ascii_digit) {
            k += 1;
        }
        k > start && (b.get(k) == Some(&end) || (end == b':' && k == b.len()))
    };
    let colon = (0..b.len()).any(|k| b.get(k) == Some(&b':') && digits_then(k + 1, b':'));
    let paren = (0..b.len()).any(|k| b.get(k) == Some(&b'(') && digits_then(k + 1, b','));
    let word = line
        .match_indices("line ")
        .any(|(k, _)| b.get(k + 5).is_some_and(u8::is_ascii_digit));
    colon || paren || word
}

fn json_values(out: &str, per_line: bool) -> Vec<serde_json::Value> {
    if per_line {
        return out
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with('{') || l.starts_with('['))
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
    }
    let start = out.find(['{', '[']).unwrap_or(out.len());
    let mut values = Vec::new();
    let stream = serde_json::Deserializer::from_str(out.get(start..).unwrap_or(""));
    for v in stream.into_iter::<serde_json::Value>() {
        match v {
            Ok(v) => values.push(v),
            Err(_) => break,
        }
    }
    values
}

/// Containers whose array members are findings.
const CONTAINERS: &[&str] = &[
    "generalDiagnostics",
    "diagnostics",
    "results",
    "issues",
    "findings",
    "comments",
    "messages",
    "offenses",
    "violations",
    "errors",
];

/// Collect the diagnostic objects in one JSON value, across the shapes the
/// registry produces: arrays of findings, per-file records (`messages`,
/// `offenses`, `violations`, `errors`), report envelopes (`results`,
/// `diagnostics`, `generalDiagnostics`, `issues`, `findings`), SARIF
/// `runs[].results`, PHPStan's `files` map, and rustc/cargo records.
fn collect_diagnostics<'v>(v: &'v serde_json::Value, out: &mut Vec<&'v serde_json::Value>) {
    use serde_json::Value;
    match v {
        Value::Array(items) => {
            for item in items {
                collect_diagnostics(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(Value::Array(runs)) = map.get("runs") {
                for run in runs {
                    if let Some(Value::Array(results)) = run.get("results") {
                        out.extend(results.iter());
                    }
                }
                return;
            }
            if let Some(reason) = map.get("reason").and_then(Value::as_str) {
                if reason == "compiler-message" {
                    out.push(v);
                }
                return;
            }
            if let Some(spans) = map.get("spans").and_then(Value::as_array) {
                let level = map.get("level").and_then(Value::as_str).unwrap_or("");
                let text = map.get("message").and_then(Value::as_str).unwrap_or("");
                let summary = text.starts_with("aborting due to")
                    || text.contains("warning emitted")
                    || text.contains("warnings emitted");
                if !summary && (!spans.is_empty() || level == "error") {
                    out.push(v);
                }
                return;
            }
            if let Some(Value::Object(files)) = map.get("files") {
                for file in files.values() {
                    collect_diagnostics(file, out);
                }
                if let Some(Value::Array(errors)) = map.get("errors") {
                    out.extend(errors.iter());
                }
                return;
            }
            if let Some(Value::Array(files)) = map.get("files") {
                for file in files {
                    collect_diagnostics(file, out);
                }
                return;
            }
            let mut container = false;
            for key in CONTAINERS {
                if let Some(Value::Array(items)) = map.get(*key) {
                    container = true;
                    for item in items {
                        collect_diagnostics(item, out);
                    }
                }
            }
            if !container && !map.contains_key("summary") && !map.contains_key("totals") {
                out.push(v);
            }
        }
        Value::String(_) => out.push(v),
        _ => {}
    }
}

fn message_of(v: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    if let Value::String(s) = v {
        return Some(truncate_message(s));
    }
    let text = [
        "message",
        "Message",
        "description",
        "msg",
        "text",
        "hint",
        "reason",
    ]
    .iter()
    .find_map(|k| match v.get(*k) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(m)) => m.get("text").and_then(Value::as_str).map(str::to_string),
        _ => None,
    })?;
    let code = [
        "code",
        "ruleId",
        "rule",
        "RuleName",
        "check_code",
        "symbol",
        "cop_name",
        "test_id",
    ]
    .iter()
    .find_map(|k| match v.get(*k) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(m)) => m.get("code").and_then(Value::as_str).map(str::to_string),
        _ => None,
    });
    Some(truncate_message(&match code {
        Some(c) => format!("[{c}] {text}"),
        None => text,
    }))
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

    /// A fresh, empty root for private directories, unique per test.
    fn scratch_root(tag: &str) -> PathBuf {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("unifur-lint-test-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn assert_empty_and_remove(root: &Path) {
        let left: Vec<_> = std::fs::read_dir(root).unwrap().collect();
        assert!(left.is_empty(), "private directories left behind: {left:?}");
        std::fs::remove_dir(root).unwrap();
    }

    const fn stand_in(
        program: &'static str,
        args: &'static [&'static str],
        input: Input,
        output: Output,
    ) -> Linter {
        Linter {
            lang: Lang::Shell,
            tool: "stand-in",
            program,
            args,
            input,
            output,
            stream: Stream::Stdout,
            executes_code: false,
            support: &[],
            env: &[],
            notes: "test stand-in",
        }
    }

    #[test]
    fn test_registry_covers_languages_and_never_uses_a_shell() {
        let shells = [
            "sh", "bash", "zsh", "dash", "ksh", "fish", "cmd", "cmd.exe", "busybox", "env",
        ];
        for l in registry() {
            assert!(
                !shells.contains(&l.program),
                "{} runs through a shell",
                l.tool
            );
            assert!(
                !(l.program == "cargo" && l.args.first() == Some(&"clippy")),
                "cargo clippy runs build scripts"
            );
            assert!(l.program != "cargo", "{}: cargo runs build scripts", l.tool);
            for a in l.args {
                let without = a.replace("<file>", "").replace("<name>", "");
                assert!(
                    !without.contains("<file") && !without.contains("<name"),
                    "{}: malformed placeholder {a}",
                    l.tool
                );
            }
            if let Input::Stdin {
                filename_arg: Some(flag),
                ..
            } = l.input
            {
                let at = l.args.iter().position(|a| *a == flag).unwrap();
                assert_eq!(
                    l.args.get(at + 1),
                    Some(&"<name>"),
                    "{}: {flag} must carry <name>",
                    l.tool
                );
            }
            if l.program == "pwsh" {
                let command = l.args.last().unwrap();
                assert!(
                    command.contains("./snippet.ps1"),
                    "pwsh may name only the generated file"
                );
            }
            assert!(!l.notes.is_empty(), "{} needs notes", l.tool);
        }
        let expected = [
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
            Lang::Haskell,
            Lang::Elixir,
            Lang::Clojure,
            Lang::Zig,
            Lang::Nim,
            Lang::Solidity,
            Lang::Shell,
            Lang::PowerShell,
            Lang::Sql,
            Lang::Terraform,
            Lang::Dockerfile,
            Lang::Yaml,
            Lang::CiConfig,
        ];
        for lang in expected {
            assert!(
                for_lang(lang).next().is_some(),
                "{} has no linter",
                lang.as_str()
            );
        }
        let clippy = registry().iter().find(|l| l.tool == "clippy").unwrap();
        assert_eq!(clippy.program, "clippy-driver");
        assert!(clippy.args.contains(&"clippy::all"));
    }

    #[test]
    fn test_missing_tool_is_unavailable_never_clean() {
        let ghost = stand_in(
            "unifur-no-such-linter-7f3a",
            &[],
            Input::TempFile { ext: "txt" },
            Output::LineCount,
        );
        assert!(!available(&ghost));
        let err = run(&ghost, "x", Duration::from_secs(1)).unwrap_err();
        assert!(err.to_string().contains("not installed"), "{err}");
        let outcomes = lint_with([ghost].iter(), "x", &RunOptions::default());
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, LintStatus::Unavailable);
        assert!(!outcomes[0].clean());
        // Every registered linter that is absent here reports the same way.
        for l in registry().iter().filter(|l| !available(l)) {
            let o = &lint_with(std::iter::once(l), "x", &RunOptions::default())[0];
            assert_eq!(o.status, LintStatus::Unavailable, "{}", l.tool);
            assert!(!o.clean());
        }
    }

    #[test]
    fn test_executing_tool_is_refused_without_opt_in() {
        let mut exec = stand_in(
            "true",
            &[],
            Input::TempFile { ext: "txt" },
            Output::ExitCodeOnly,
        );
        exec.executes_code = true;
        if available(&exec) {
            let err = run(&exec, "x", Duration::from_secs(5)).unwrap_err();
            assert!(err.to_string().contains("refusing"), "{err}");
            let o = &lint_with(std::iter::once(&exec), "x", &RunOptions::default())[0];
            assert_eq!(o.status, LintStatus::Refused);
            assert!(!o.clean());
            let allowed = RunOptions {
                allow_executing: true,
                ..RunOptions::default()
            };
            assert!(run_with(&exec, "x", &allowed).unwrap().clean());
        } else {
            assert_eq!(
                lint_with(std::iter::once(&exec), "x", &RunOptions::default())[0].status,
                LintStatus::Unavailable
            );
        }
    }

    #[test]
    fn test_stdin_round_trip_and_cleanup() {
        let root = scratch_root("cat");
        let cat = stand_in(
            "cat",
            &[],
            Input::Stdin {
                filename_arg: None,
                ext: "txt",
            },
            Output::LineCount,
        );
        let opts = RunOptions {
            temp_root: Some(root.clone()),
            ..RunOptions::default()
        };
        if available(&cat) {
            let big = "line\n".repeat(50_000);
            let o = run_with(&cat, &big, &opts).unwrap();
            assert_eq!(o.status, LintStatus::Completed);
            assert_eq!(o.diagnostics, 50_000);
            assert_eq!(o.messages.len(), opts.max_messages);
            let empty = run_with(&cat, "", &opts).unwrap();
            assert!(empty.clean());
        } else {
            assert!(run_with(&cat, "x", &opts).is_err());
        }
        assert_empty_and_remove(&root);
    }

    #[test]
    fn test_temp_file_is_private_and_removed_even_on_error() {
        let root = scratch_root("ls");
        let opts = RunOptions {
            temp_root: Some(root.clone()),
            ..RunOptions::default()
        };
        let ls = stand_in(
            "ls",
            &["-l", "<file>"],
            Input::TempFile { ext: "txt" },
            Output::LineCount,
        );
        if available(&ls) {
            let o = run_with(&ls, "secret", &opts).unwrap();
            assert_eq!(o.diagnostics, 1);
            #[cfg(unix)]
            assert!(o.messages[0].starts_with("-rw-------"), "{:?}", o.messages);
        }
        // A tool whose output cannot be read is an error, and the directory
        // still goes away.
        let unreadable = stand_in(
            "ls",
            &["<file>"],
            Input::TempFile { ext: "txt" },
            Output::Json,
        );
        if available(&unreadable) {
            let err = run_with(&unreadable, "x", &opts).unwrap_err();
            assert!(err.to_string().contains("refusing to score"), "{err}");
        }
        assert_empty_and_remove(&root);
    }

    #[test]
    fn test_timeout_kills_the_child() {
        let root = scratch_root("sleep");
        let sleeper = stand_in(
            "sleep",
            &["30"],
            Input::TempFile { ext: "txt" },
            Output::ExitCodeOnly,
        );
        let opts = RunOptions {
            timeout: Duration::from_millis(300),
            temp_root: Some(root.clone()),
            ..RunOptions::default()
        };
        if available(&sleeper) {
            let start = Instant::now();
            let o = run_with(&sleeper, "x", &opts).unwrap();
            assert!(o.timed_out);
            assert_eq!(o.status, LintStatus::TimedOut);
            assert!(!o.clean());
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "took {:?}",
                start.elapsed()
            );
        } else {
            assert_eq!(
                lint_with(std::iter::once(&sleeper), "x", &opts)[0].status,
                LintStatus::Unavailable
            );
        }
        assert_empty_and_remove(&root);
    }

    #[test]
    fn test_environment_is_minimal_and_home_is_private() {
        let dir = Path::new("/nonexistent/private");
        let env = child_env(dir, &[("EXTRA", "1")]);
        let get = |k: &str| {
            env.iter()
                .rev()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("HOME"), Some(dir.as_os_str().to_os_string()));
        assert_eq!(
            get("XDG_CONFIG_HOME"),
            Some(dir.join(".config").into_os_string())
        );
        assert_eq!(get("EXTRA"), Some("1".into()));
        let allowed = [
            "PATH",
            "HOME",
            "TMPDIR",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "LC_ALL",
            "NO_COLOR",
            "TERM",
            "EXTRA",
        ];
        for (name, _) in &env {
            let name = name.to_string_lossy();
            assert!(
                allowed.contains(&name.as_ref()) || LOCATION_VARS.contains(&name.as_ref()),
                "unexpected variable {name}"
            );
        }
        if let Some(path) = get("PATH") {
            assert!(std::env::split_paths(&path).all(|p| p.is_absolute()));
        }
    }

    #[test]
    fn test_output_shapes_are_counted() {
        let l = |output| stand_in("x", &[], Input::TempFile { ext: "x" }, output);
        let ruff = r#"[{"code":"F401","message":"`os` imported but unused","fix":{"edits":[{"content":""}]}}]"#;
        assert_eq!(parse_output(&l(Output::Json), ruff, false).unwrap().0, 1);
        let eslint = r#"[{"filePath":"a.js","messages":[{"ruleId":"no-unused-vars","message":"x"},{"ruleId":"eqeqeq","message":"y"}]}]"#;
        assert_eq!(parse_output(&l(Output::Json), eslint, false).unwrap().0, 2);
        let pyright = r#"{"version":"1","generalDiagnostics":[{"message":"m","rule":"reportX"}],"summary":{"errorCount":1}}"#;
        assert_eq!(parse_output(&l(Output::Json), pyright, false).unwrap().0, 1);
        let rubocop = r#"{"files":[{"path":"a.rb","offenses":[{"cop_name":"Style/X","message":"m"}]}],"summary":{"offense_count":1}}"#;
        assert_eq!(parse_output(&l(Output::Json), rubocop, false).unwrap().0, 1);
        let phpstan = r#"{"totals":{"errors":0,"file_errors":2},"files":{"a.php":{"errors":2,"messages":[{"message":"a"},{"message":"b"}]}},"errors":[]}"#;
        assert_eq!(parse_output(&l(Output::Json), phpstan, false).unwrap().0, 2);
        let sarif =
            r#"{"version":"2.1.0","runs":[{"results":[{"ruleId":"R1","message":{"text":"t"}}]}]}"#;
        let (n, msgs) = parse_output(&l(Output::Sarif), sarif, false).unwrap();
        assert_eq!(n, 1);
        assert_eq!(msgs, vec!["[R1] t".to_string()]);
        let rustc = concat!(
            r#"{"$message_type":"diagnostic","message":"unused variable: `x`","level":"warning","spans":[{"line_start":1}]}"#,
            "\n",
            r#"{"$message_type":"diagnostic","message":"1 warning emitted","level":"warning","spans":[]}"#,
            "\n"
        );
        assert_eq!(
            parse_output(&l(Output::JsonLines), rustc, true).unwrap().0,
            1
        );
        assert_eq!(parse_output(&l(Output::Json), "[]", true).unwrap().0, 0);
        assert!(parse_output(&l(Output::Json), "", true).is_err());
        assert!(parse_output(
            &l(Output::Json),
            "Traceback (most recent call last):",
            false
        )
        .is_err());
        assert_eq!(parse_output(&l(Output::JsonLines), "", true).unwrap().0, 0);
        assert!(parse_output(&l(Output::JsonLines), "", false).is_err());
        let located = "# snippet\n./snippet.go:5:2: x declared and not used\nsnippet.ts(3,7): error TS2322\nfoo at line 4, column 1.\nchatter\n";
        assert_eq!(
            parse_output(&l(Output::Located), located, false).unwrap().0,
            3
        );
        assert!(parse_output(&l(Output::Located), "panic: boom\n", false).is_err());
        assert_eq!(
            parse_output(&l(Output::ExitCodeOnly), "Errors parsing x", false)
                .unwrap()
                .0,
            1
        );
        assert_eq!(
            parse_output(&l(Output::ExitCodeOnly), "No syntax errors", true)
                .unwrap()
                .0,
            0
        );
    }

    /// Real linters, when installed: a known-bad snippet must never come back
    /// as a clean pass, and rustc (present wherever this crate builds) must
    /// actually report.
    #[test]
    fn test_installed_linters_never_pass_a_bad_snippet() {
        let bad: &[(Lang, &str)] = &[
            (Lang::Rust, "pub fn f() { let x = 1; }\n"),
            (
                Lang::Go,
                "package snippet\n\nimport \"fmt\"\n\nfunc F() { fmt.Printf(\"%d\", \"s\") }\n",
            ),
            (Lang::TypeScript, "const n: number = \"s\";\n"),
            (Lang::Php, "<?php\nfunction f( {\n"),
            (Lang::Ruby, "def f\n  x = 1\nend\n"),
            (Lang::JavaScript, "function f( {\n"),
            (
                Lang::Python,
                "import os\n\nvalue: int = \"text\"\nassert value\n",
            ),
            (Lang::Shell, "rm $1\n"),
            (Lang::PowerShell, "Write-Host 'x'\n"),
            (Lang::CSharp, "class C { void M() { int x; } }\n"),
            (Lang::Perl, "my $x = 1;\n"),
        ];
        let opts = RunOptions {
            timeout: Duration::from_secs(120),
            ..RunOptions::default()
        };
        for (lang, source) in bad {
            for l in for_lang(*lang) {
                if l.tool == "gofmt" {
                    continue; // formatting, not correctness
                }
                let o = &lint_with(std::iter::once(l), source, &opts)[0];
                eprintln!(
                    "{:>14} {:?}: {} diagnostic(s) {:?}",
                    l.tool,
                    o.status,
                    o.diagnostics,
                    o.messages.first()
                );
                assert!(!o.clean(), "{} passed a known-bad snippet", l.tool);
                if l.tool == "rustc" && available(l) {
                    assert_eq!(o.status, LintStatus::Completed, "{:?}", o.messages);
                    assert!(o.diagnostics >= 1);
                }
            }
        }
        if let Some(rustc) = for_lang(Lang::Rust).find(|l| l.tool == "rustc") {
            if available(rustc) {
                let ok = run_with(rustc, "pub fn f() -> u32 { 1 }\n", &opts).unwrap();
                assert!(ok.clean(), "{ok:?}");
            }
        }
    }
}
