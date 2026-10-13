//! Benchmark harness for the NINA specialist fleet.
//!
//! Loads one staged `.mpk` checkpoint and scores it on standard language
//! benchmarks with a byte-level scorer. Multiple-choice tasks (ARC, HellaSwag,
//! WinoGrande, PIQA, LAMBADA) are scored by summed continuation
//! log-likelihood, the lm-eval-harness convention; generative tasks (GSM8K,
//! HumanEval) are greedy-decoded and checked.
//!
//! Every task reports the number of items actually scored, because a subset
//! score is not a benchmark score and the reader has to be able to tell which
//! one they are looking at. The random-chance baseline is printed next to each
//! number for the same reason: an accuracy below chance is the finding, and it
//! is invisible without the baseline beside it.
//!
//! `--backend wgpu` runs the same scoring on the GPU. The harness is generic
//! over the backend so the two paths cannot drift: the CPU build is the
//! reference, the GPU build is the fast one, and they score the same tensors.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use diffusionblocks::checkpoint;
use diffusionblocks::student::{GeometricStudent, StudentConfig};
use diffusionblocks::tokenizer::ByteTokenizer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One scored item and the metric it feeds.
#[derive(Clone, Debug)]
struct Item {
    /// Context bytes; the choice is scored as its continuation.
    ctx: String,
    /// Candidate continuations, in dataset order.
    choices: Vec<String>,
    /// Index of the correct choice.
    gold: usize,
}

/// Decode one forward pass into `[seq, vocab]` logits.
fn logits_of<B: Backend<FloatElem = f32>>(
    model: &GeometricStudent<B>,
    ids: &[u16],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let flat: Vec<i64> = ids.iter().map(|t| i64::from(*t)).collect();
    let t = Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape([1, ids.len()]);
    let logits = model.forward(t)?.logits;
    logits
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| format!("logits to f32: {e}").into())
}

/// Greedy-decode `max_new` tokens after `prompt`, stopping early at a stop
/// sequence. No KV cache: each step re-runs the prefix, which is what makes the
/// CPU path slow and the GPU path worth enabling.
fn greedy<B: Backend<FloatElem = f32>>(
    model: &GeometricStudent<B>,
    prompt: &str,
    max_new: usize,
    stop: &[&str],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<String, Box<dyn std::error::Error>> {
    let tok = ByteTokenizer::new();
    let ctx = model.config().context;
    let mut ids: Vec<u16> = tok.encode(prompt);
    if ids.is_empty() {
        ids.push(b' ' as u16);
    }
    if ids.len() >= ctx {
        ids.truncate(ctx - 1);
    }
    let mut out: Vec<u16> = Vec::new();
    for _ in 0..max_new {
        let data = logits_of(model, &ids, device)?;
        let vocab = 259usize;
        let n = ids.len();
        let last = &data[(n - 1) * vocab..n * vocab];
        let mut arg = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (j, v) in last.iter().enumerate() {
            if *v > best {
                best = *v;
                arg = j;
            }
        }
        // 0 and 258 are the tokenizer's specials; neither is a continuation.
        if arg == 0 || arg == 258 {
            break;
        }
        let next = u16::try_from(arg)?;
        ids.push(next);
        out.push(next);
        if out.len() % 16 == 0 {
            let s = tok.decode_lossy(&out);
            if stop.iter().any(|p| s.contains(p)) {
                break;
            }
        }
    }
    Ok(tok.decode_lossy(&out))
}

/// Count positions where `argmax(logits)` matches the continuation byte, and
/// return `(summed log p, per-byte mean log p, token hits)`.
fn score_choice<B: Backend<FloatElem = f32>>(
    model: &GeometricStudent<B>,
    ctx: &str,
    cont: &str,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<(f64, f64, usize), Box<dyn std::error::Error>> {
    let tok = ByteTokenizer::new();
    let c: Vec<u16> = tok.encode(ctx);
    let k: Vec<u16> = tok.encode(cont);
    if k.is_empty() {
        return Ok((f64::NEG_INFINITY, f64::NEG_INFINITY, 0));
    }
    let ctx_len = c.len();
    let mut ids = c;
    ids.extend_from_slice(&k);
    let data = logits_of(model, &ids, device)?;
    let vocab = 259usize;
    let mut total = 0.0f64;
    let mut hits = 0usize;
    // Position t predicts token t+1; the continuation occupies ctx_len..ids.len().
    for i in ctx_len..ids.len() {
        let row = &data[i * vocab..(i + 1) * vocab];
        let mut mx = f32::NEG_INFINITY;
        for v in row.iter() {
            if *v > mx {
                mx = *v;
            }
        }
        let sumexp: f32 = row.iter().map(|v| (*v - mx).exp()).sum();
        let gold = k[i - ctx_len] as usize;
        total += f64::from(row[gold] - mx - sumexp.ln());
        let mut arg = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (j, v) in row.iter().enumerate() {
            if *v > best {
                best = *v;
                arg = j;
            }
        }
        if arg == gold {
            hits += 1;
        }
    }
    let n_cont = k.len();
    Ok((total, total / n_cont as f64, hits))
}

fn load_jsonl(p: &Path) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(p)?;
    let mut out = Vec::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(l)?;
        out.push(v);
    }
    Ok(out)
}

/// Parse a JSON array of strings, or fail with the field named.
fn str_array(v: &serde_json::Value, key: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    v.get(key)
        .and_then(|a| a.as_array())
        .ok_or_else(|| format!("field '{key}' is not an array").into())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect::<Vec<String>>()
        })
}

fn load_tasks(dir: &Path, want: &str) -> Result<Vec<(String, Item)>, Box<dyn std::error::Error>> {
    let mut out: Vec<(String, Item)> = Vec::new();
    let want: Vec<&str> = want.split(',').collect();
    for (task, file) in [
        ("arc_easy", "arc_easy.jsonl"),
        ("arc_challenge", "arc_challenge.jsonl"),
        ("hellaswag", "hellaswag.jsonl"),
        ("winogrande", "winogrande.jsonl"),
        ("piqa", "piqa.jsonl"),
        ("lambada", "lambada.jsonl"),
    ] {
        if !want.contains(&"all") && !want.contains(&task) {
            continue;
        }
        let p = dir.join(file);
        if !p.exists() {
            eprintln!("skip {task}: {} not present", p.display());
            continue;
        }
        let rows = load_jsonl(&p)?;
        let before = out.len();
        for v in &rows {
            let it = match task {
                "arc_easy" | "arc_challenge" => {
                    let choices = v.get("choices").ok_or("arc: no 'choices' object")?;
                    let labels = str_array(choices, "label")?;
                    let texts = str_array(choices, "text")?;
                    if labels.len() != texts.len() || labels.is_empty() {
                        return Err(format!(
                            "arc: {} labels vs {} texts",
                            labels.len(),
                            texts.len()
                        )
                        .into());
                    }
                    let akey = v["answerKey"].as_str().unwrap_or("");
                    let gold = labels
                        .iter()
                        .position(|l| l == akey)
                        .ok_or("arc: answerKey not among labels")?;
                    Item {
                        ctx: format!(
                            "Question: {}\nAnswer:",
                            v["question"].as_str().unwrap_or("")
                        ),
                        choices: texts,
                        gold,
                    }
                }
                "hellaswag" => {
                    let ctx = v["ctx"].as_str().unwrap_or("").to_string();
                    let act = v["activity_label"].as_str().unwrap_or("");
                    let endings = str_array(v, "endings")?;
                    let gold = v["label"]
                        .as_str()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    Item {
                        ctx: format!("{act}: {ctx}"),
                        choices: endings,
                        gold,
                    }
                }
                "winogrande" => {
                    let s = v["sentence"].as_str().unwrap_or("");
                    let o1 = v["option1"].as_str().unwrap_or("");
                    let o2 = v["option2"].as_str().unwrap_or("");
                    let idx = s.find('_').ok_or("winogrande: no blank")?;
                    let pre = &s[..idx];
                    let post = &s[idx + 1..];
                    let gold = if v["answer"].as_str() == Some("1") {
                        0
                    } else {
                        1
                    };
                    Item {
                        ctx: pre.to_string(),
                        choices: vec![format!("{o1}{post}"), format!("{o2}{post}")],
                        gold,
                    }
                }
                "piqa" => {
                    let goal = v["goal"].as_str().unwrap_or("");
                    let sol1 = v["sol1"].as_str().unwrap_or("");
                    let sol2 = v["sol2"].as_str().unwrap_or("");
                    let gold = v["label"].as_u64().unwrap_or(0) as usize;
                    Item {
                        ctx: format!("Goal: {goal}\nSolution:"),
                        choices: vec![format!(" {sol1}"), format!(" {sol2}")],
                        gold,
                    }
                }
                "lambada" => {
                    let text = v["text"].as_str().unwrap_or("");
                    let t = text.rsplit_once(' ').ok_or("lambada: no space")?;
                    Item {
                        ctx: t.0.to_string(),
                        choices: vec![format!(" {}", t.1)],
                        gold: 0,
                    }
                }
                _ => continue,
            };
            out.push((task.to_string(), it));
        }
        eprintln!("loaded {task}: {} items", out.len() - before);
    }
    Ok(out)
}

struct Args {
    ckpt: PathBuf,
    boxes: usize,
    dir: PathBuf,
    tasks: Vec<String>,
    limit: usize,
    genlimit: usize,
    maxnew: usize,
    backend: String,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        ckpt: PathBuf::new(),
        boxes: 10,
        dir: PathBuf::from("/tmp/opencode/bench"),
        tasks: vec!["all".to_string()],
        limit: 400,
        genlimit: 25,
        maxnew: 64,
        backend: "cpu".to_string(),
    };
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let need = |i: &mut usize| -> Result<String, Box<dyn std::error::Error>> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value").into())
        };
        match flag {
            "--ckpt" => a.ckpt = PathBuf::from(need(&mut i)?),
            "--boxes" => a.boxes = need(&mut i)?.parse()?,
            "--dir" => a.dir = PathBuf::from(need(&mut i)?),
            "--task" => a.tasks.push(need(&mut i)?),
            "--limit" => a.limit = need(&mut i)?.parse()?,
            "--genlimit" => a.genlimit = need(&mut i)?.parse()?,
            "--maxnew" => a.maxnew = need(&mut i)?.parse()?,
            "--backend" => a.backend = need(&mut i)?,
            other => return Err(format!("unknown flag {other}").into()),
        }
        i += 1;
    }
    Ok(a)
}

/// Build the box layout the checkpoint's MoSME width demands.
///
/// Box count is `coding_boxes.len() + field_boxes.len()`. The label order is a
/// permutation that leaves the forward pass invariant — the router column and
/// the expert block it selects move together — so canonical names rebuild the
/// exact graph the weights were trained in.
fn config_for(boxes: usize) -> Result<StudentConfig, Box<dyn std::error::Error>> {
    const CODING: [&str; 10] = [
        "rust", "python", "c", "cpp", "jsts", "web", "go", "java", "csharp", "php",
    ];
    const FIELDS: [&str; 6] = ["math", "science", "bio", "cyber", "tech", "pentest"];
    let (nc, nf) = match boxes {
        10 => (10, 0),
        11 => (6, 5),
        12 => (6, 6),
        other => return Err(format!("unsupported box count {other}").into()),
    };
    let config = StudentConfig {
        coding_boxes: CODING[..nc].iter().map(|s| (*s).to_string()).collect(),
        field_boxes: FIELDS[..nf].iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    config.validate()?;
    Ok(config)
}

fn run<B: Backend<FloatElem = f32>>(
    a: &Args,
    device: <B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = config_for(a.boxes)?;
    <B as Backend>::seed(&device, 0);
    eprintln!("building model (boxes={}, backend) ...", a.boxes);
    let model = GeometricStudent::<B>::new(&config, &device)?;
    diffusionblocks::tensor_ext::force_initialization(&model);
    let model = checkpoint::load::<B, _>(model, &a.ckpt, &device)?;
    eprintln!("loaded {}", a.ckpt.display());

    let wants = |t: &str| a.tasks.iter().any(|x| x == "all" || x == t);

    // ---- generative: GSM8K and HumanEval ----
    if wants("gsm8k") {
        let p = a.dir.join("gsm8k.jsonl");
        if p.exists() {
            let rows = load_jsonl(&p)?;
            let n = a.genlimit.min(rows.len());
            let mut correct = 0usize;
            for (idx, v) in rows.iter().take(n).enumerate() {
                let q = v["question"].as_str().unwrap_or("");
                let gold = v["answer"]
                    .as_str()
                    .unwrap_or("")
                    .split("####")
                    .nth(1)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let out = greedy(&model, &format!("{q}\nAnswer:"), a.maxnew, &["\n\n"], &device)?;
                let pred: String = out
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                if !gold.is_empty() && pred == gold {
                    correct += 1;
                }
                if idx % 25 == 0 {
                    eprintln!("  gsm8k {idx}/{n} correct={correct}");
                }
            }
            println!(
                "{}",
                serde_json::json!({"task":"gsm8k","n":n,"correct":correct,
                                   "accuracy": correct as f64/n as f64,
                                   "random_baseline": 0.0})
            );
        }
    }

    if wants("humaneval") {
        let p = a.dir.join("humaneval.jsonl");
        if p.exists() {
            let rows = load_jsonl(&p)?;
            let n = a.genlimit.min(rows.len());
            let mut solved = 0usize;
            for (idx, v) in rows.iter().take(n).enumerate() {
                let prompt = v["prompt"].as_str().unwrap_or("");
                let out = greedy(
                    &model,
                    prompt,
                    a.maxnew,
                    &["\ndef ", "\nclass ", "\nif __name__", "\nprint("],
                    &device,
                )?;
                let full = format!("{prompt}{out}");
                let test = v["test"].as_str().unwrap_or("");
                let entry = v["entry_point"].as_str().unwrap_or("");
                let tmp = format!("/tmp/opencode/he_{}.py", std::process::id());
                std::fs::write(&tmp, format!("{full}\n\n{test}\n\ncheck({entry})\n"))?;
                let ok = std::process::Command::new("python3")
                    .arg("-c")
                    .arg(
                        "import sys,resource;resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));\
                         import runpy;runpy.run_path(sys.argv[1],run_name='__main__')",
                    )
                    .arg(&tmp)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if ok {
                    solved += 1;
                }
                if idx % 10 == 0 {
                    eprintln!("  humaneval {idx}/{n} solved={solved}");
                }
            }
            println!(
                "{}",
                serde_json::json!({"task":"humaneval","n":n,"solved":solved,
                                   "pass@1": solved as f64/n as f64,
                                   "random_baseline": 0.0})
            );
        }
    }

    // ---- multiple choice by continuation log-likelihood ----
    let want = if a.tasks.iter().any(|t| t == "all") {
        String::from("all")
    } else {
        a.tasks.join(",")
    };
    let items = load_tasks(&a.dir, &want)?;
    let items = if items.len() > a.limit {
        items[..a.limit].to_vec()
    } else {
        items
    };

    // (n, acc_hits, acc_norm_hits) per task. LAMBADA is graded separately
    // because its single continuation is exact-token, not argmax-over-choices.
    let mut by_task: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut lam = (0usize, 0usize, 0usize, 0usize);

    for (idx, (tname, it)) in items.iter().enumerate() {
        if it.choices.len() == 1 {
            let (_, _, hits) = score_choice(&model, &it.ctx, &it.choices[0], &device)?;
            let n_tok = ByteTokenizer::new().encode(&it.choices[0]).len();
            lam.0 += 1;
            lam.1 += usize::from(hits == n_tok);
            lam.2 += hits;
            lam.3 += n_tok;
        } else {
            let mut best_sum = f64::NEG_INFINITY;
            let mut best_norm = f64::NEG_INFINITY;
            let mut arg_sum = 0usize;
            let mut arg_norm = 0usize;
            for (ci, ch) in it.choices.iter().enumerate() {
                let (s, nrm, _) = score_choice(&model, &it.ctx, ch, &device)?;
                if s > best_sum {
                    best_sum = s;
                    arg_sum = ci;
                }
                if nrm > best_norm {
                    best_norm = nrm;
                    arg_norm = ci;
                }
            }
            let e = by_task.entry(tname.clone()).or_default();
            e.0 += 1;
            e.1 += usize::from(arg_sum == it.gold);
            e.2 += usize::from(arg_norm == it.gold);
        }
        if (idx + 1) % 50 == 0 {
            eprintln!("  {idx}/{} scored", items.len());
        }
    }

    if lam.0 > 0 {
        println!(
            "{}",
            serde_json::json!({"task":"lambada","n":lam.0,
                "acc_exact": lam.1 as f64/lam.0 as f64,
                "acc_token": lam.2 as f64/lam.3 as f64,
                "random_baseline": 0.0})
        );
    }

    let mut keys: Vec<String> = by_task.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let (n, c, cn) = by_task[&k];
        let chance = match k.as_str() {
            "winogrande" | "piqa" => 0.5,
            _ => 0.25,
        };
        println!(
            "{}",
            serde_json::json!({"task":k,"n":n,
                "acc": c as f64/n as f64,
                "acc_norm": cn as f64/n as f64,
                "random_baseline": chance})
        );
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
fn run_wgpu(a: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, Wgpu, WgpuDevice};
    let device = WgpuDevice::DiscreteGpu(0);
    let setup =
        std::panic::catch_unwind(|| init_setup::<AutoGraphicsApi>(&device, Default::default()))
            .map_err(|_| "WGPU initialization panicked")?;
    let info = setup.adapter.get_info();
    eprintln!(
        "backend=wgpu name={:?} type={:?} driver={:?}",
        info.name, info.device_type, info.driver
    );
    run::<Wgpu>(a, device)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args()?;
    match a.backend.as_str() {
        "cpu" => run::<burn::backend::NdArray<f32>>(&a, Default::default()),
        #[cfg(feature = "wgpu")]
        "wgpu" => run_wgpu(&a),
        other => Err(format!("unknown backend '{other}' (want cpu or wgpu)").into()),
    }
}
