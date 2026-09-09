//! Hyperparameter sweeps as a reproducible protocol (roadmap Phase 28;
//! issue #1 §5, §8, §14).
//!
//! A grid is a list of axes, each a flag name and its values; every cell is
//! trained once per seed with the same configuration otherwise, and the
//! result is one [`Record`] per cell whose trials are the per-seed final
//! losses. Nothing is compared across cells that differ in more than the grid
//! says, and every raw number is kept. The command exists so that the GPU
//! comparisons the roadmap lists as blocked are a single invocation when the
//! hardware appears -- the protocol is the part that can be written now.

use anyhow::Context;
use std::path::Path;

use crate::experiment::{Record, RunLog};
use crate::train::{train, Objective, TrainConfig};

/// One axis of a grid: a flag and the values it sweeps over.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    pub key: String,
    pub values: Vec<String>,
}

/// The whole grid.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Grid {
    pub axes: Vec<Axis>,
}

impl Grid {
    /// `lr=1e-4,3e-4 num_blocks=2,3` -- axes separated by whitespace or `;`,
    /// values by `,`.
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let mut axes = Vec::new();
        for item in spec.split(|c: char| c.is_whitespace() || c == ';').filter(|s| !s.is_empty()) {
            let (key, values) = item
                .split_once('=')
                .with_context(|| format!("grid axis {item:?} is not key=v1,v2"))?;
            let values: Vec<String> = values.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect();
            anyhow::ensure!(!values.is_empty(), "grid axis {key:?} has no values");
            anyhow::ensure!(SUPPORTED.contains(&key), "unknown grid key {key:?}; supported: {}", SUPPORTED.join(", "));
            axes.push(Axis { key: key.to_string(), values });
        }
        anyhow::ensure!(!axes.is_empty(), "an empty grid sweeps nothing");
        Ok(Self { axes })
    }

    /// Every combination, in lexicographic order of the axes.
    pub fn cells(&self) -> Vec<Vec<(String, String)>> {
        let mut cells: Vec<Vec<(String, String)>> = vec![Vec::new()];
        for axis in &self.axes {
            let mut next = Vec::with_capacity(cells.len() * axis.values.len());
            for cell in &cells {
                for value in &axis.values {
                    let mut c = cell.clone();
                    c.push((axis.key.clone(), value.clone()));
                    next.push(c);
                }
            }
            cells = next;
        }
        cells
    }
}

/// Keys [`apply`] understands.
pub const SUPPORTED: [&str; 19] = [
    "lr", "steps", "batch_size", "num_blocks", "gamma", "weight_decay", "accumulate", "clip_norm",
    "ema_decay", "uncertainty", "importance_bins", "normalize_block_loss", "objective", "consistency",
    "lr_schedule", "moe_experts", "moe_top_k", "balance_scope", "bias_balance_rate",
];

/// Set one flag on a configuration.
pub fn apply(config: &mut TrainConfig, key: &str, value: &str) -> anyhow::Result<()> {
    let num = |v: &str| -> anyhow::Result<f64> { v.parse::<f64>().with_context(|| format!("{key}={value}: not a number")) };
    let int = |v: &str| -> anyhow::Result<usize> { v.parse::<usize>().with_context(|| format!("{key}={value}: not an integer")) };
    match key {
        "lr" => config.lr = num(value)?,
        "steps" => config.steps = int(value)?,
        "batch_size" => config.batch_size = int(value)?,
        "num_blocks" => config.num_blocks = int(value)?,
        "gamma" => config.gamma = num(value)?,
        "weight_decay" => config.weight_decay = num(value)?,
        "accumulate" => config.accumulate = int(value)?.max(1),
        "clip_norm" => config.clip_norm = (num(value)? > 0.0).then(|| num(value).unwrap_or(0.0) as f32),
        "ema_decay" => config.ema_decay = (num(value)? > 0.0).then(|| num(value).unwrap_or(0.0)),
        "uncertainty" => config.uncertainty = num(value)?,
        "importance_bins" => config.importance_bins = int(value)?,
        "normalize_block_loss" => config.normalize_block_loss = matches!(value, "1" | "true" | "yes"),
        "objective" => config.objective = Objective::parse(value)?,
        "consistency" => {
            let weight = num(value)?;
            config.objective = if weight > 0.0 {
                let cfg = crate::consistency::ConsistencyConfig {
                    schedule: crate::consistency::ConsistencySchedule::Constant { weight },
                    ..Default::default()
                };
                Objective::Consistency(Box::new(cfg))
            } else {
                Objective::Dblock
            };
        }
        "lr_schedule" => config.lr_schedule = crate::schedule::LrSchedule::parse(value, config.lr, config.steps)?,
        "moe_experts" => {
            let mut moe = config.moe.unwrap_or_default();
            moe.num_experts = int(value)?;
            config.moe = Some(moe);
        }
        "moe_top_k" => {
            let mut moe = config.moe.unwrap_or_default();
            moe.top_k = int(value)?;
            config.moe = Some(moe);
        }
        "balance_scope" => config.balance_scope = crate::schedule::BalanceScope::parse(value)?,
        "bias_balance_rate" => {
            config.bias_balance_rate = num(value)? as f32;
            if let Some(moe) = config.moe.as_mut() {
                moe.balance_bias = config.bias_balance_rate > 0.0;
            }
        }
        other => anyhow::bail!("unknown grid key {other:?}"),
    }
    Ok(())
}

/// A cell's name, e.g. `sweep/lr=3e-4,num_blocks=2`.
pub fn cell_name(cell: &[(String, String)]) -> String {
    let parts: Vec<String> = cell.iter().map(|(k, v)| format!("{k}={v}")).collect();
    format!("sweep/{}", parts.join(","))
}

/// Train one cell once per seed and summarize the final losses.
///
/// The learning-rate schedule is re-derived after the overrides so a swept
/// `lr` or `steps` reaches it. Every per-seed outcome is kept in `extra`.
pub fn run_cell(base: &TrainConfig, cell: &[(String, String)], seeds: &[u64]) -> anyhow::Result<Record> {
    let mut config = base.clone();
    for (key, value) in cell {
        apply(&mut config, key, value)?;
    }
    if !cell.iter().any(|(k, _)| k == "lr_schedule") {
        config.lr_schedule =
            crate::schedule::LrSchedule::parse(config.lr_schedule.name(), config.lr, config.steps)?;
    }
    let config_json = serde_json::to_value(&config).context("serialize sweep cell config")?;
    let mut record = Record::new(cell_name(cell), "loss", config_json, seeds.to_vec());
    let mut per_seed = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let run = TrainConfig { seed, ..config.clone() };
        let (_, summary) = train(&run)?;
        let per_block: Vec<f32> = (0..summary.health.num_blocks())
            .map(|b| summary.health.block(b).map_or(0.0, |h| h.mean_loss()))
            .collect();
        record.push(f64::from(summary.final_loss), false);
        per_seed.push(serde_json::json!({
            "seed": seed,
            "final_loss": summary.final_loss,
            "mean_loss": summary.mean_loss,
            "steps_taken": summary.steps_taken,
            "steps_skipped": summary.steps_skipped,
            "elapsed_secs": summary.elapsed_secs,
            "per_block_mean_loss": per_block,
            "aborted": summary.aborted,
        }));
    }
    record.extra = serde_json::json!({ "per_seed": per_seed, "cell": cell });
    Ok(record)
}

/// Run every cell and append each record to `log` as soon as it exists, so
/// a sweep interrupted halfway still leaves its finished cells on disk.
pub fn run_grid(base: &TrainConfig, grid: &Grid, seeds: &[u64], log: Option<&Path>) -> anyhow::Result<Vec<Record>> {
    let mut records = Vec::new();
    for cell in grid.cells() {
        let record = run_cell(base, &cell, seeds)?;
        if let Some(path) = log {
            RunLog::append(path, &record)?;
        }
        records.push(record);
    }
    Ok(records)
}

/// A table of a sweep's records.
pub fn render(records: &[Record]) -> String {
    let mut out = format!("{:<48} {:>5} {:>12} {:>12} {:>12}\n", "cell", "seeds", "mean loss", "±ci95", "median");
    out.push_str(&"-".repeat(94));
    out.push('\n');
    for r in records {
        match r.summary {
            Some(s) => out.push_str(&format!(
                "{:<48} {:>5} {:>12.4} {:>12.4} {:>12.4}\n",
                r.name,
                s.n,
                s.mean,
                if s.ci95_half_width.is_nan() { 0.0 } else { s.ci95_half_width },
                s.median
            )),
            None => out.push_str(&format!("{:<48} {:>5} {:>12} {:>12} {:>12}\n", r.name, 0, "-", "-", "-")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_parses_and_enumerates_every_cell() {
        let grid = Grid::parse("lr=1e-4,3e-4 num_blocks=2,3;consistency=0").unwrap();
        assert_eq!(grid.axes.len(), 3);
        let cells = grid.cells();
        assert_eq!(cells.len(), 4);
        assert_eq!(cell_name(&cells[0]), "sweep/lr=1e-4,num_blocks=2,consistency=0");
        assert_eq!(cell_name(&cells[3]), "sweep/lr=3e-4,num_blocks=3,consistency=0");
        assert!(Grid::parse("lr=").is_err());
        assert!(Grid::parse("bogus=1").is_err());
        assert!(Grid::parse("").is_err());
    }

    #[test]
    fn test_apply_sets_the_flag_it_names() {
        let mut config = TrainConfig::default();
        apply(&mut config, "lr", "5e-4").unwrap();
        apply(&mut config, "consistency", "0.3").unwrap();
        apply(&mut config, "moe_experts", "6").unwrap();
        apply(&mut config, "bias_balance_rate", "1e-3").unwrap();
        assert_eq!(config.lr, 5e-4);
        assert!(matches!(config.objective, Objective::Consistency(_)));
        let moe = config.moe.unwrap();
        assert_eq!(moe.num_experts, 6);
        assert!(moe.balance_bias);
        apply(&mut config, "consistency", "0").unwrap();
        assert!(matches!(config.objective, Objective::Dblock));
        assert!(apply(&mut config, "lr", "fast").is_err());
    }
}
