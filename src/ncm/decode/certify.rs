//! The `ncm-decode` certificate group: the decoding strategies' mechanics,
//! stated against the doubles in `doubles.rs`. Every certificate carries a
//! control that must fail (or must be non-vacuous) for the certificate to
//! pass, so a check that cannot see a broken decoder is itself a failure.

use anyhow::{anyhow, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::doubles::{CountingCodec, CountingModel, ToyCodec, ToyModel};
use super::{
    draw_index, speculative_code, CodeSampler, DecodeBudget, DecodeConfig, DecodeOutput, Decoder,
    EarlyExit, OutlineCache, OutlineCacheUse, Particles, Repair, Speculation,
};
use crate::ncm::{ConceptCodec, LevelModel};
use crate::verify::Certificate;

type B = NdArray<f32>;
type Device = NdArrayDevice;

const GROUP: &str = "ncm-decode";
const LENGTHS: [usize; 4] = [1, 2, 4, 8];

fn certificate(
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate {
        group: GROUP,
        name,
        theorem,
        residual,
        tolerance,
    }
}

/// Residual of a check whose control must hold: the measured residual if
/// it does, infinity if it does not.
fn controlled(residual: f64, control_holds: bool) -> f64 {
    if control_holds {
        residual
    } else {
        f64::INFINITY
    }
}

/// Every check in the group.
pub(crate) fn certificate_checks() -> Result<Vec<Certificate>> {
    let device = Device::default();
    Ok(vec![
        speculative_sampling_preserves_the_target_distribution(&device)?,
        greedy_speculation_equals_plain_greedy(&device)?,
        draft_equal_to_target_accepts_everything(&device)?,
        early_exit_disabled_is_plain_decoding(&device)?,
        particles_one_is_plain_sampling(&device)?,
        repair_disabled_is_identity(&device)?,
        budget_is_never_exceeded(&device)?,
        decode_report_counts_calls(&device)?,
    ])
}

fn mismatched_rows(a: &[Vec<usize>], b: &[Vec<usize>]) -> usize {
    a.len().abs_diff(b.len())
        + a.iter()
            .zip(b)
            .map(|(x, y)| {
                x.len().abs_diff(y.len()) + x.iter().zip(y).filter(|(p, q)| p != q).count()
            })
            .sum::<usize>()
}

/// Differing codes and tokens between two outputs (0 iff bit-identical).
fn mismatches(a: &DecodeOutput, b: &DecodeOutput) -> usize {
    mismatched_rows(&a.tokens, &b.tokens)
        + a.codes.len().abs_diff(b.codes.len())
        + a.codes
            .iter()
            .zip(&b.codes)
            .map(|(x, y)| mismatched_rows(x, y))
            .sum::<usize>()
}

fn total_variation(counts: &[usize], p: &[f64]) -> f64 {
    let n: usize = counts.iter().sum();
    let n = n.max(1) as f64;
    0.5 * counts
        .iter()
        .zip(p)
        .map(|(&c, &q)| (c as f64 / n - q).abs())
        .sum::<f64>()
}

fn exact_tv(a: &[f64], b: &[f64]) -> f64 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

fn toy_codec(seed: u64, device: &Device) -> Result<ToyCodec<B>> {
    ToyCodec::new(LENGTHS.to_vec(), 8, 4, 6, 3.0, seed, device)
}

fn toy_target(seed: u64, batched: bool, device: &Device) -> Result<ToyModel<B>> {
    Ok(ToyModel::random(LENGTHS.to_vec(), 8, 4, 1.0, seed + 1000, device)?.batching(batched))
}

fn toy(seed: u64, device: &Device) -> Result<(ToyCodec<B>, ToyModel<B>)> {
    Ok((toy_codec(seed, device)?, toy_target(seed, false, device)?))
}

fn prompt(batch: usize, seed: u64, device: &Device) -> Tensor<B, 2, Int> {
    let tokens: Vec<i64> = (0..batch * 2)
        .map(|i| ((i as u64 * 7 + seed * 3) % 6) as i64)
        .collect();
    Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device).reshape([batch, 2])
}

/// The exact joint sampling distribution of (level-0 code, level-1 codes) of
/// a 2-level schedule with one level-0 code, indexed `c0 · C^ℓ₁` plus the
/// level-1 codes read as a base-`C` number (first position most significant).
fn exact_two_level_joint(
    codec: &ToyCodec<B>,
    model: &dyn LevelModel<B>,
    sampler: &CodeSampler,
    device: &Device,
) -> Result<Vec<f64>> {
    let codebook = codec.codebook_size();
    let lengths = codec.schedule().lengths().to_vec();
    let (first, second) = match lengths.as_slice() {
        [a, b] => (*a, *b),
        other => {
            return Err(anyhow!(
                "the joint certificate needs a 2-level schedule, got {other:?}"
            ))
        }
    };
    anyhow::ensure!(first == 1, "level 0 must have one code, has {first}");
    let host =
        |t: Tensor<B, 3>| -> Vec<f32> { t.into_data().convert::<f32>().iter::<f32>().collect() };
    let mut p0 = Vec::new();
    let zero = codec.reconstruct(&[], 1)?;
    sampler.distribution(&host(model.level_logits(0, zero, None)?), &mut p0);
    let block = codebook
        .checked_pow(u32::try_from(second)?)
        .ok_or_else(|| anyhow!("{codebook}^{second} level-1 outcomes overflow"))?;
    let mut joint = vec![0.0; codebook * block];
    let mut dist = Vec::new();
    for (c0, &w0) in p0.iter().enumerate() {
        let code =
            Tensor::<B, 1, Int>::from_ints([i64::try_from(c0)?].as_slice(), device).reshape([1, 1]);
        let zhat = codec.reconstruct(&[code], 1)?;
        let logits = host(model.level_logits(1, zhat, None)?);
        let per_position: Vec<Vec<f64>> = logits
            .chunks_exact(codebook)
            .map(|position| {
                sampler.distribution(position, &mut dist);
                dist.clone()
            })
            .collect();
        // Enumerate every level-1 assignment as a base-C number.
        for (rest, slot) in joint.iter_mut().skip(c0 * block).take(block).enumerate() {
            let mut probability = w0;
            let mut digits = rest;
            for position in per_position.iter().rev() {
                probability *= position.get(digits % codebook).copied().unwrap_or(0.0);
                digits /= codebook;
            }
            *slot = probability;
        }
    }
    Ok(joint)
}

fn speculative_sampling_preserves_the_target_distribution(device: &Device) -> Result<Certificate> {
    const TOLERANCE: f64 = 0.04;

    // (a) One code, known p and q: the correct rule against two wrong ones.
    let p = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25];
    let q = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05];
    let draws = 40_000;
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let mut residual = Vec::new();
    let (mut right, mut wrong, mut lazy) = ([0usize; 6], [0usize; 6], [0usize; 6]);
    for _ in 0..draws {
        let drafted = draw_index(&q, rng.random::<f64>());
        let u = rng.random::<f64>();
        let (code, _) = speculative_code(&p, &q, drafted, u, &mut rng, &mut residual);
        if let Some(slot) = right.get_mut(code) {
            *slot += 1;
        }
        // Wrong rule: the ratio inverted, min(1, q/p).
        let (pd, qd) = (
            p.get(drafted).copied().unwrap_or(0.0),
            q.get(drafted).copied().unwrap_or(0.0),
        );
        let inverted = if u < (qd / pd).min(1.0) {
            drafted
        } else {
            let rest: Vec<f64> = p.iter().zip(&q).map(|(a, b)| (a - b).max(0.0)).collect();
            draw_index(&rest, rng.random::<f64>())
        };
        if let Some(slot) = wrong.get_mut(inverted) {
            *slot += 1;
        }
        // Lazier still: accept every draft.
        if let Some(slot) = lazy.get_mut(drafted) {
            *slot += 1;
        }
    }
    let tv_single = total_variation(&right, &p);
    let single_controls =
        total_variation(&wrong, &p) > TOLERANCE && total_variation(&lazy, &p) > TOLERANCE;

    // (b) Through the decoder: two levels, level 1 conditioned on level 0,
    // batched verification, temperature and top-k, many sequences at once.
    let codec = ToyCodec::<B>::new(vec![1, 2], 4, 3, 4, 2.0, 5, device)?;
    let target = ToyModel::<B>::random(vec![1, 2], 4, 3, 1.5, 6, device)?.batching(true);
    let draft = target.perturbed(1.0, 7);
    let sampler = CodeSampler {
        temperature: 0.9,
        top_k: Some(3),
    };
    let exact = exact_two_level_joint(&codec, &target, &sampler, device)?;
    let draft_exact = exact_two_level_joint(&codec, &draft, &sampler, device)?;
    let codebook = codec.codebook_size();
    let decoder = Decoder::new(&codec, &target, device).with_draft(&draft);
    let mut counts = vec![0usize; exact.len()];
    let (mut accepted, mut rejected, mut discarded, mut full_spans) = (0, 0, 0, 0);
    // Many small batches: a batch shares one level loop, so with few rows a
    // fully accepted level 0 (whose drafted codes condition the verified
    // level 1) happens often enough to be exercised, not just the discard.
    let batch = 16;
    for seed in 0..1280 {
        let config = DecodeConfig {
            sampler,
            seed,
            speculation: Some(Speculation { draft_levels: 2 }),
            ..DecodeConfig::default()
        };
        let out = decoder.decode(None, batch, &config, &DecodeBudget::unlimited(), None)?;
        accepted += out.report.accepted_draft_codes;
        rejected += out.report.rejected_draft_codes;
        discarded += out.report.discarded_drafts;
        // One step drafted both levels and nothing was thrown away: level 1
        // was emitted from the verification conditioned on drafted level 0.
        full_spans += usize::from(out.report.draft_calls == 2 && out.report.discarded_drafts == 0);
        let (level0, level1) = match out.codes.as_slice() {
            [a, b] => (a, b),
            _ => {
                return Err(anyhow!(
                    "a 2-level decode returned {} levels",
                    out.codes.len()
                ))
            }
        };
        for (c0, rest) in level0.iter().zip(level1) {
            let index = rest
                .iter()
                .fold(c0.first().copied().unwrap_or(0), |acc, &c| {
                    acc * codebook + c
                });
            if let Some(slot) = counts.get_mut(index) {
                *slot += 1;
            }
        }
    }
    let tv_joint = total_variation(&counts, &exact);
    // An always-accept decoder would emit the draft's joint.
    let joint_controls = exact_tv(&draft_exact, &exact) > 2.5 * TOLERANCE
        && accepted > 0
        && rejected > 0
        && discarded > 0
        && full_spans > 0;

    Ok(certificate(
        "speculative_sampling_preserves_the_target_distribution",
        "Speculative sampling emits the target's distribution: over 40000 seeded single-code draws (known p, q) and \
         20480 decoded sequences of a 2-level toy whose level 1 depends on level 0 (perturbed draft, T=0.9, top-3, \
         batched verification, both the accepted-span and the discard path taken) the empirical distribution is \
         within total variation 0.04 of the exact target; the \
         inverted rule min(1,q/p), an always-accept rule, and a decoder emitting the draft's joint all exceed it.",
        controlled(tv_single.max(tv_joint), single_controls && joint_controls),
        TOLERANCE,
    ))
}

fn greedy_speculation_equals_plain_greedy(device: &Device) -> Result<Certificate> {
    let mut differences = 0usize;
    let (mut accepted, mut rejected, mut control_differs) = (0usize, 0usize, false);
    for seed in 0..4u64 {
        let (codec, target) = toy(seed, device)?;
        let draft = target.perturbed(0.8, seed + 50);
        let plain_config = DecodeConfig {
            seed,
            ..DecodeConfig::default()
        };
        let budget = DecodeBudget::unlimited();
        let plain = Decoder::new(&codec, &target, device).decode(
            Some(prompt(3, seed, device)),
            3,
            &plain_config,
            &budget,
            None,
        )?;
        let draft_plain = Decoder::new(&codec, &draft, device).decode(
            Some(prompt(3, seed, device)),
            3,
            &plain_config,
            &budget,
            None,
        )?;
        control_differs |= mismatches(&plain, &draft_plain) > 0;
        for batched in [false, true] {
            let target = toy_target(seed, batched, device)?;
            for draft_levels in 1..=4 {
                let config = DecodeConfig {
                    speculation: Some(Speculation { draft_levels }),
                    ..plain_config.clone()
                };
                let spec = Decoder::new(&codec, &target, device)
                    .with_draft(&draft)
                    .decode(Some(prompt(3, seed, device)), 3, &config, &budget, None)?;
                differences += mismatches(&plain, &spec);
                accepted += spec.report.accepted_drafts;
                rejected += spec.report.rejected_drafts;
            }
        }
    }
    Ok(certificate(
        "greedy_speculation_equals_plain_greedy",
        "Greedy level speculation (accept a code iff draft and target argmax agree) emits exactly plain greedy target \
         decoding, codes and tokens bit for bit, for draft spans 1-4, batched or not; drafts were both accepted and \
         rejected, and the draft's own greedy decode differs from the target's.",
        controlled(differences as f64, accepted > 0 && rejected > 0 && control_differs),
        0.0,
    ))
}

fn draft_equal_to_target_accepts_everything(device: &Device) -> Result<Certificate> {
    let mut misses = 0usize;
    let mut control_rejects = 0usize;
    for seed in 0..3u64 {
        let (codec, target) = toy(seed, device)?;
        let other = target.perturbed(0.8, seed + 60);
        for sampler in [
            CodeSampler::greedy(),
            CodeSampler::with_temperature(1.0),
            CodeSampler {
                temperature: 0.7,
                top_k: Some(3),
            },
        ] {
            let config = DecodeConfig {
                sampler,
                seed,
                speculation: Some(Speculation { draft_levels: 3 }),
                ..DecodeConfig::default()
            };
            let budget = DecodeBudget::unlimited();
            let out = Decoder::new(&codec, &target, device)
                .with_draft(&target)
                .decode(None, 4, &config, &budget, None)?;
            misses += out.report.rejected_draft_codes
                + out.report.discarded_drafts
                + LENGTHS.len().abs_diff(out.report.accepted_drafts);
            let control = Decoder::new(&codec, &target, device)
                .with_draft(&other)
                .decode(None, 4, &config, &budget, None)?;
            control_rejects += control.report.rejected_draft_codes;
        }
    }
    Ok(certificate(
        "draft_equal_to_target_accepts_everything",
        "With the target as its own draft every drafted code is accepted (p(d)/q(d) = 1), so all K levels come from \
         accepted drafts with nothing rejected or discarded, greedy or sampled; a perturbed draft is rejected somewhere.",
        controlled(misses as f64, control_rejects > 0),
        0.0,
    ))
}

fn early_exit_disabled_is_plain_decoding(device: &Device) -> Result<Certificate> {
    let mut differences = 0usize;
    let (mut probes, mut control_exits, mut runs) = (0usize, 0usize, 0usize);
    for seed in 0..4u64 {
        let (codec, target) = toy(seed, device)?;
        let decoder = Decoder::new(&codec, &target, device);
        for sampler in [CodeSampler::greedy(), CodeSampler::with_temperature(1.0)] {
            let plain_config = DecodeConfig {
                sampler,
                seed,
                ..DecodeConfig::default()
            };
            let budget = DecodeBudget::unlimited();
            let plain = decoder.decode(None, 2, &plain_config, &budget, None)?;
            let never = DecodeConfig {
                early_exit: Some(EarlyExit {
                    threshold: 1.5,
                    min_levels: 1,
                }),
                ..plain_config.clone()
            };
            let out = decoder.decode(None, 2, &never, &budget, None)?;
            differences +=
                mismatches(&plain, &out) + usize::from(out.report.early_exit_level.is_some());
            probes += out.report.probes;
            let always = DecodeConfig {
                early_exit: Some(EarlyExit {
                    threshold: 0.0,
                    min_levels: 1,
                }),
                ..plain_config
            };
            let control = decoder.decode(None, 2, &always, &budget, None)?;
            control_exits += usize::from(
                control.report.early_exit_level == Some(1) && control.report.levels_used == 1,
            );
            runs += 1;
        }
    }
    Ok(certificate(
        "early_exit_disabled_is_plain_decoding",
        "Early exit at threshold > 1 never fires and reproduces plain decoding exactly (codes and tokens), although \
         its probe runs after every level; at threshold 0 it always exits at the minimum level.",
        controlled(differences as f64, probes > 0 && control_exits == runs),
        0.0,
    ))
}

fn particles_one_is_plain_sampling(device: &Device) -> Result<Certificate> {
    let mut differences = 0usize;
    let (mut control_differs, mut resamples) = (false, 0usize);
    for seed in 0..6u64 {
        let (codec, target) = toy(seed, device)?;
        let decoder = Decoder::new(&codec, &target, device);
        let sampler = if seed % 2 == 0 {
            CodeSampler::with_temperature(1.0)
        } else {
            CodeSampler {
                temperature: 0.7,
                top_k: Some(3),
            }
        };
        let plain_config = DecodeConfig {
            sampler,
            seed,
            ..DecodeConfig::default()
        };
        let budget = DecodeBudget::unlimited();
        let plain = decoder.decode(
            Some(prompt(2, seed, device)),
            2,
            &plain_config,
            &budget,
            None,
        )?;
        let with = |count| DecodeConfig {
            particles: Some(Particles {
                count,
                tilt: 1.0,
                ess_threshold: 1.0,
            }),
            ..plain_config.clone()
        };
        let one = decoder.decode(Some(prompt(2, seed, device)), 2, &with(1), &budget, None)?;
        differences += mismatches(&plain, &one);
        // Three requested, one allowed: the budget's clamp is plain sampling too.
        let clamp = DecodeBudget {
            max_particles: 1,
            ..budget
        };
        let clamped = decoder.decode(Some(prompt(2, seed, device)), 2, &with(3), &clamp, None)?;
        differences += mismatches(&plain, &clamped) + clamped.report.particles.abs_diff(1);
        let three = decoder.decode(Some(prompt(2, seed, device)), 2, &with(3), &budget, None)?;
        control_differs |= mismatches(&plain, &three) > 0;
        resamples += three.report.resamples;
    }
    Ok(certificate(
        "particles_one_is_plain_sampling",
        "A single particle (requested, or clamped by the budget) is plain sampling with the same seed, bit for bit; \
         three particles resample and change the output.",
        controlled(differences as f64, control_differs && resamples > 0),
        0.0,
    ))
}

fn repair_disabled_is_identity(device: &Device) -> Result<Certificate> {
    let mut differences = 0usize;
    let (mut checks, mut repaired, mut control_differs) = (0usize, 0usize, false);
    for seed in 0..6u64 {
        let (codec, target) = toy(seed, device)?;
        let decoder = Decoder::new(&codec, &target, device);
        let plain_config = DecodeConfig {
            sampler: CodeSampler::with_temperature(1.0),
            seed,
            ..DecodeConfig::default()
        };
        let budget = DecodeBudget::unlimited();
        let plain = decoder.decode(None, 2, &plain_config, &budget, None)?;
        let repair = |guard| DecodeConfig {
            repair: Some(Repair {
                confidence_guard: guard,
                min_alternative_probability: 0.0,
            }),
            ..plain_config.clone()
        };
        // Guard 0 flags nothing; no repairs or no probes allowed does nothing.
        let guarded = decoder.decode(None, 2, &repair(0.0), &budget, None)?;
        checks += guarded.report.repair_checks;
        differences += mismatches(&plain, &guarded);
        let no_repairs = DecodeBudget {
            max_repairs: 0,
            ..budget
        };
        differences += mismatches(
            &plain,
            &decoder.decode(None, 2, &repair(2.0), &no_repairs, None)?,
        );
        let no_probes = DecodeBudget {
            max_probes: 0,
            ..budget
        };
        differences += mismatches(
            &plain,
            &decoder.decode(None, 2, &repair(2.0), &no_probes, None)?,
        );
        let control = decoder.decode(None, 2, &repair(2.0), &budget, None)?;
        repaired += control.report.repaired_positions;
        control_differs |= mismatches(&plain, &control) > 0;
    }
    Ok(certificate(
        "repair_disabled_is_identity",
        "Immune repair that flags nothing (guard 0), may not repair (max_repairs 0) or may not probe (max_probes 0) \
         leaves plain decoding bit for bit, although the guard-0 check runs; an unguarded repair rewrites codes.",
        controlled(differences as f64, checks > 0 && repaired > 0 && control_differs),
        0.0,
    ))
}

fn random_config(rng: &mut ChaCha8Rng, seed: u64) -> DecodeConfig {
    let levels = LENGTHS.len();
    let sampler = match rng.random_range(0..3) {
        0 => CodeSampler::greedy(),
        1 => CodeSampler::with_temperature(rng.random_range(0.5..1.5)),
        _ => CodeSampler {
            temperature: rng.random_range(0.5..1.5),
            top_k: Some(rng.random_range(1..6)),
        },
    };
    DecodeConfig {
        sampler,
        seed,
        early_exit: rng.random_bool(0.5).then(|| EarlyExit {
            threshold: rng.random_range(0.2..1.2),
            min_levels: rng.random_range(1..=levels),
        }),
        speculation: rng.random_bool(0.5).then(|| Speculation {
            draft_levels: rng.random_range(1..5),
        }),
        repair: rng.random_bool(0.5).then(|| Repair {
            confidence_guard: rng.random_range(0.0..2.0),
            min_alternative_probability: rng.random_range(0.0..0.5),
        }),
        particles: rng.random_bool(0.5).then(|| Particles {
            count: rng.random_range(1..5),
            tilt: rng.random_range(0.0..2.0),
            ess_threshold: rng.random_range(0.0..1.0),
        }),
        outline_cache: rng.random_bool(0.5).then(|| OutlineCacheUse {
            levels: rng.random_range(1..=levels),
            bypass_probability: rng.random_range(0.0..1.0),
            model_id: "toy".into(),
        }),
    }
}

fn random_budget(rng: &mut ChaCha8Rng) -> DecodeBudget {
    DecodeBudget {
        max_level_calls: rng.random_range(0..10),
        max_draft_calls: rng.random_range(0..6),
        max_repairs: rng.random_range(0..3),
        max_particles: rng.random_range(1..5),
        max_probes: rng.random_range(0..6),
    }
}

fn budget_is_never_exceeded(device: &Device) -> Result<Certificate> {
    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let (mut violations, mut binding, mut decodes) = (0usize, 0usize, 0usize);
    for seed in 0..48u64 {
        let codec = toy_codec(seed % 4, device)?;
        let target = toy_target(seed % 4, rng.random_bool(0.5), device)?;
        let draft = target.perturbed(0.8, seed);
        let config = random_config(&mut rng, seed);
        let budget = random_budget(&mut rng);
        let mut cache = OutlineCache::new(8)?;
        // Twice, so an outline cache can hit on the second pass.
        for _ in 0..2 {
            let counted_target = CountingModel::new(&target);
            let counted_draft = CountingModel::new(&draft);
            let decoder = Decoder::new(&codec, &counted_target, device).with_draft(&counted_draft);
            match decoder.decode(
                Some(prompt(2, seed, device)),
                2,
                &config,
                &budget,
                Some(&mut cache),
            ) {
                Ok(out) => {
                    decodes += 1;
                    violations += usize::from(budget.admits(&out.report).is_err())
                        + usize::from(counted_target.passes() > budget.max_level_calls)
                        + usize::from(counted_draft.passes() > budget.max_draft_calls);
                }
                // The one refusal a budget may cause: no target call at all
                // and nothing cached.
                Err(err) => {
                    violations += usize::from(
                        budget.max_level_calls > 0 || !err.to_string().contains("max_level_calls"),
                    );
                }
            }
        }
        // Control: the same config unbudgeted would have overspent.
        let mut fresh = OutlineCache::new(8)?;
        let unlimited = Decoder::new(&codec, &target, device)
            .with_draft(&draft)
            .decode(
                Some(prompt(2, seed, device)),
                2,
                &config,
                &DecodeBudget::unlimited(),
                Some(&mut fresh),
            )?;
        binding += usize::from(budget.admits(&unlimited.report).is_err());
    }
    Ok(certificate(
        "budget_is_never_exceeded",
        "Over 48 random configurations with every strategy randomly on (twice each, sharing an outline cache) and \
         random budgets, no decode spends more target calls, draft calls, repairs, particles or probes than its \
         budget — by its report and by independent counters — and the only refusal is a zero call budget; the same \
         configurations unbudgeted overspend at least 10 times.",
        controlled(violations as f64, binding >= 10 && decodes >= 48),
        0.0,
    ))
}

fn decode_report_counts_calls(device: &Device) -> Result<Certificate> {
    let (codec, plain_target) = toy(9, device)?;
    let batched_target = toy_target(9, true, device)?;
    let draft = plain_target.perturbed(0.8, 90);
    let sampled = DecodeConfig {
        sampler: CodeSampler::with_temperature(1.0),
        seed: 9,
        ..DecodeConfig::default()
    };
    let spec = DecodeConfig {
        speculation: Some(Speculation { draft_levels: 3 }),
        ..sampled.clone()
    };
    let everything = DecodeConfig {
        speculation: Some(Speculation { draft_levels: 2 }),
        early_exit: Some(EarlyExit {
            threshold: 0.9,
            min_levels: 2,
        }),
        repair: Some(Repair {
            confidence_guard: 1.5,
            min_alternative_probability: 0.0,
        }),
        particles: Some(Particles {
            count: 3,
            tilt: 0.5,
            ess_threshold: 0.8,
        }),
        outline_cache: Some(OutlineCacheUse {
            levels: 1,
            bypass_probability: 0.0,
            model_id: "toy".into(),
        }),
        ..sampled.clone()
    };
    let cases: [(&ToyModel<B>, &DecodeConfig, usize); 4] = [
        (&plain_target, &sampled, 1),
        (&plain_target, &spec, 1),
        (&batched_target, &spec, 1),
        (&batched_target, &everything, 2),
    ];
    let mut discrepancy = 0usize;
    let mut levels_vs_passes = Vec::new();
    let mut cache = OutlineCache::new(4)?;
    for (target, config, repeats) in cases {
        for _ in 0..repeats {
            let counted_target = CountingModel::new(target);
            let counted_draft = CountingModel::new(&draft);
            let counted_codec = CountingCodec::new(&codec);
            let out = Decoder::new(&counted_codec, &counted_target, device)
                .with_draft(&counted_draft)
                .decode(
                    Some(prompt(2, 9, device)),
                    2,
                    config,
                    &DecodeBudget::unlimited(),
                    Some(&mut cache),
                )?;
            let r = &out.report;
            let (reconstructs, decodes, encodes) = counted_codec.counts();
            discrepancy += r.level_calls.abs_diff(counted_target.passes())
                + r.batched_calls
                    .abs_diff(counted_target.multi_level_passes())
                + r.level_evaluations.abs_diff(counted_target.levels())
                + r.draft_calls.abs_diff(counted_draft.passes())
                + r.reconstructs.abs_diff(reconstructs)
                + r.decodes.abs_diff(decodes)
                + r.encodes.abs_diff(encodes);
            levels_vs_passes.push((
                counted_target.levels(),
                counted_target.passes(),
                r.levels_used,
                r.cached_levels,
            ));
        }
    }
    // Controls: the non-batching target paid at least one pass per level
    // (speculation saved it nothing), and the batching one evaluated more
    // levels than it made passes — so counting levels instead of passes, or
    // the reverse, would show up as a discrepancy.
    let honest_unbatched = matches!(levels_vs_passes.get(1), Some(&(levels, passes, used, _)) if levels == passes && passes >= used);
    let batching_seen =
        matches!(levels_vs_passes.get(2), Some(&(levels, passes, _, _)) if levels > passes);
    let cache_seen = matches!(levels_vs_passes.get(4), Some(&(_, _, _, cached)) if cached == 1);
    Ok(certificate(
        "decode_report_counts_calls",
        "The decode report's target passes, batched passes, level evaluations, draft passes, reconstructs, decodes and \
         encodes equal independent counters wrapped around the models and codec — plain, speculative against a \
         non-batching target (one pass per level, no saving claimed), speculative against a batching one (fewer passes \
         than levels), and every strategy at once with a cache hit.",
        controlled(discrepancy as f64, honest_unbatched && batching_seen && cache_seen),
        0.0,
    ))
}
