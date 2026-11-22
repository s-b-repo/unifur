//! Exact signed readouts over quantized attention blocks.

use super::{floats, Attention, Heads, Linear, Matrix, HEAD_SIZE};
use anyhow::{ensure, Result};
use burn::tensor::activation::softmax;
use serde::Serialize;
use std::cmp::Reverse;

const LOGIT_LIMIT: i128 = 16;
// 64² * sqrt(64) * ln(2), rounded to the nearest integer.
const SCORE_DIVISOR: i128 = 22_713;
const FEATURE_SCALE: f64 = 64.0;
const EVIDENCE_SCALE: f64 = 65_536.0;
const BLOCK_SIZE: usize = 8;

type Interval = (i128, i128);

struct Block {
    keys: Vec<i16>,
    margins: Vec<i32>,
    ranges: Vec<(i16, i16)>,
    negative: i128,
    positive: i128,
}

fn dot(query: &[i16], key: &[i16]) -> i128 {
    query
        .iter()
        .zip(key)
        .map(|(&q, &k)| i128::from(q) * i128::from(k))
        .sum()
}

fn mass(score: i128, divisor: i128) -> i128 {
    let exponent = score.div_euclid(divisor).clamp(-LOGIT_LIMIT, LOGIT_LIMIT) + LOGIT_LIMIT;
    1 << exponent
}

impl Block {
    fn new(keys: &[i16], margins: &[i32], width: usize) -> Self {
        let mut ranges = vec![(i16::MAX, i16::MIN); width];
        for row in keys.chunks_exact(width) {
            for ((low, high), &key) in ranges.iter_mut().zip(row) {
                *low = (*low).min(key);
                *high = (*high).max(key);
            }
        }
        let (mut negative, mut positive) = (0, 0);
        for &margin in margins {
            negative += i128::from(margin.min(0));
            positive += i128::from(margin.max(0));
        }
        Self {
            keys: keys.to_vec(),
            margins: margins.to_vec(),
            ranges,
            negative,
            positive,
        }
    }

    fn bound(&self, query: &[i16], divisor: i128) -> Interval {
        let (mut low, mut high) = (0, 0);
        for (&weight, &(mut min, mut max)) in query.iter().zip(&self.ranges) {
            if weight < 0 {
                std::mem::swap(&mut min, &mut max);
            }
            low += i128::from(weight) * i128::from(min);
            high += i128::from(weight) * i128::from(max);
        }
        let light = mass(low, divisor);
        let heavy = mass(high, divisor);
        (
            light * self.positive + heavy * self.negative,
            heavy * self.positive + light * self.negative,
        )
    }

    fn score(&self, query: &[i16], divisor: i128) -> i128 {
        self.keys
            .chunks_exact(query.len())
            .zip(&self.margins)
            .map(|(key, &margin)| mass(dot(query, key), divisor) * i128::from(margin))
            .sum()
    }
}

struct Readout {
    blocks: Vec<Block>,
    width: usize,
    tokens: usize,
}

struct Decision {
    interval: Interval,
    scored: usize,
}

impl Decision {
    fn positive(&self) -> bool {
        self.interval.0 > 0
    }
}

impl Readout {
    fn new(keys: &[i16], margins: &[i32], width: usize, block_size: usize) -> Result<Self> {
        ensure!(width > 0 && block_size > 0, "invalid attention block shape");
        ensure!(
            !margins.is_empty() && margins.len().checked_mul(width) == Some(keys.len()),
            "key and margin counts differ"
        );
        let size = block_size
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("attention block size overflow"))?;
        Ok(Self {
            blocks: keys
                .chunks(size)
                .zip(margins.chunks(block_size))
                .map(|(key, margin)| Block::new(key, margin, width))
                .collect(),
            width,
            tokens: margins.len(),
        })
    }

    fn decide(&self, query: &[i16], divisor: i128, certified: bool) -> Result<Decision> {
        ensure!(divisor > 0, "attention score divisor must be positive");
        ensure!(query.len() == self.width, "query and key widths differ");
        // Each weighted i32 term has magnitude at most 2^63. Input allocations on
        // a 64-bit host have fewer than 2^63 terms, including bounds and updates.
        if !certified {
            let score = self
                .blocks
                .iter()
                .map(|block| block.score(query, divisor))
                .sum();
            return Ok(Decision {
                interval: (score, score),
                scored: self.tokens,
            });
        }
        let (mut lower, mut upper) = (0, 0);
        let mut pending = Vec::new();
        for block in &self.blocks {
            let (low, high) = block.bound(query, divisor);
            lower += low;
            upper += high;
            if low != high {
                pending.push((low, high, block));
            }
        }
        let mut scored = 0;
        if lower <= 0 && upper > 0 {
            pending.sort_by_key(|(low, high, _)| Reverse(high - low));
            for (low, high, block) in pending {
                let exact = block.score(query, divisor);
                lower += exact - low;
                upper += exact - high;
                scored += block.margins.len();
                if lower > 0 || upper <= 0 {
                    break;
                }
            }
        }
        Ok(Decision {
            interval: (lower, upper),
            scored,
        })
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Report {
    pub(super) queries: usize,
    pub(super) total_token_scores: usize,
    pub(super) scored_tokens: usize,
    pub(super) float_agreements: usize,
    reference_value_projections: usize,
    pooled_value_projections: usize,
    max_projection_difference: f32,
}

fn feature(value: f32) -> Result<i16> {
    let quantized = (f64::from(value) * FEATURE_SCALE).round();
    ensure!(
        (f64::from(i16::MIN)..=f64::from(i16::MAX)).contains(&quantized),
        "attention feature exceeds the signed 16-bit range"
    );
    Ok(quantized as i16) // audit-allow: range and finiteness checked above
}

fn evidence(value: f64) -> Result<i32> {
    let quantized = (value * EVIDENCE_SCALE).round();
    ensure!(
        (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&quantized),
        "attention evidence exceeds the signed 32-bit range"
    );
    Ok(quantized as i32) // audit-allow: range and finiteness checked above
}

/// Compare one trained head's contribution to output coordinate zero with zero.
/// The integer readout is a diagnostic and the float context serves as its control.
pub(super) fn probe(
    attention: &Attention,
    input: Matrix,
    queries: Heads,
    keys: Heads,
    values: Heads,
) -> Result<Report> {
    let [heads, queries_per_head, _] = queries.dims();
    let tokens = keys.dims()[1];
    let weights = softmax(
        queries.clone().matmul(keys.clone().swap_dims(1, 2)) / (HEAD_SIZE as f64).sqrt(),
        2,
    );
    let readouts = floats(attention.output.weight.clone().narrow(1, 0, 1))?;
    let mut report = Report {
        queries: heads * queries_per_head,
        total_token_scores: heads * queries_per_head * tokens,
        scored_tokens: 0,
        float_agreements: 0,
        reference_value_projections: heads * tokens,
        pooled_value_projections: heads * queries_per_head,
        max_projection_difference: 0.0,
    };
    for head in 0..heads {
        let probabilities = weights.clone().narrow(0, head, 1).squeeze_dim::<2>(0);
        let head_values = values.clone().narrow(0, head, 1).squeeze_dim::<2>(0);
        let projection_column = 2 * attention.width + head * HEAD_SIZE;
        let value_projection = Linear {
            weight: attention
                .qkv
                .weight
                .clone()
                .narrow(1, projection_column, HEAD_SIZE),
            bias: attention
                .qkv
                .bias
                .as_ref()
                .map(|bias| bias.clone().narrow(1, projection_column, HEAD_SIZE)),
        };
        let baseline = probabilities.clone().matmul(head_values.clone());
        let pooled = value_projection.forward(probabilities.matmul(input.clone()));
        let difference = floats((baseline.clone() - pooled).abs().max())?[0];
        ensure!(
            difference.is_finite(),
            "non-finite attention projection residual"
        );
        report.max_projection_difference = report.max_projection_difference.max(difference);

        let readout = &readouts[head * HEAD_SIZE..(head + 1) * HEAD_SIZE];
        let margin = |row: &[f32]| {
            row.iter()
                .zip(readout)
                .map(|(&value, &weight)| f64::from(value) * f64::from(weight))
                .sum::<f64>()
        };
        let key = floats(keys.clone().narrow(0, head, 1))?
            .into_iter()
            .map(feature)
            .collect::<Result<Vec<_>>>()?;
        let margins = floats(head_values)?
            .chunks_exact(HEAD_SIZE)
            .map(|row| evidence(margin(row)))
            .collect::<Result<Vec<_>>>()?;
        let index = Readout::new(&key, &margins, HEAD_SIZE, BLOCK_SIZE)?;
        let query = floats(queries.clone().narrow(0, head, 1))?;
        for (row, context) in query
            .chunks_exact(HEAD_SIZE)
            .zip(floats(baseline)?.chunks_exact(HEAD_SIZE))
        {
            let query = row
                .iter()
                .copied()
                .map(feature)
                .collect::<Result<Vec<_>>>()?;
            let full = index.decide(&query, SCORE_DIVISOR, false)?;
            let certified = index.decide(&query, SCORE_DIVISOR, true)?;
            ensure!(
                certified.positive() == full.positive()
                    && (certified.interval.0..=certified.interval.1).contains(&full.interval.0),
                "the full integer margin escaped the attention certificate"
            );
            report.scored_tokens += certified.scored;
            report.float_agreements += usize::from(certified.positive() == (margin(context) > 0.0));
        }
    }
    Ok(report)
}

pub(super) fn residual() -> Result<f64> {
    let easy = Readout::new(&[0, 1, 0, 0], &[-1, 14, -1, 1], 1, 2)?;
    let tie = Readout::new(&[0, 0, 1], &[-1, 1, 0], 1, 3)?;
    let mixed = Readout::new(&[-3, 2, 2, 3, 4, -1, 6, 3], &[-9, 5, -3, 7], 2, 2)?;
    let mut failed = false;
    for (readout, query, divisor) in [
        (&easy, &[1][..], 1),
        (&tie, &[1], 1),
        (&mixed, &[-3, 2], 2),
        (&mixed, &[i16::MAX, i16::MIN], 1),
    ] {
        let full = readout.decide(query, divisor, false)?;
        let result = readout.decide(query, divisor, true)?;
        failed |= result.positive() != full.positive()
            || !(result.interval.0..=result.interval.1).contains(&full.interval.0);
    }
    failed |= easy.decide(&[1], 1, true)?.scored != 0;
    failed |= tie.decide(&[1], 1, true)?.scored != 3;
    Ok(f64::from(u8::from(failed)))
}
