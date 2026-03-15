# VECTOR: Multi-Objective NAS for Reservoir Computing Anomaly Detection

VECTOR finds compact, high-accuracy Echo State Network configurations for time series anomaly detection. It uses NSGA-II multi-objective search across 8 TSAD benchmarks to discover ESN architectures that balance detection accuracy (F1) against model size, producing Pareto-optimal configurations that match or exceed hand-tuned baselines.

## Installation

```bash
git clone <repo-url> && cd vector
pip install -e ".[dev]"
```

## Dataset Download

VECTOR evaluates on 8 benchmark datasets:

| Dataset | Domain | Source |
|---------|--------|--------|
| NAB | Cloud/IT metrics | Numenta Anomaly Benchmark |
| UCR | Synthetic patterns | UCR Time Series Archive |
| MBA | Medical telemetry | MIT-BIH Arrhythmia |
| SMAP | Spacecraft telemetry | NASA SMAP soil moisture |
| MSL | Spacecraft telemetry | NASA Mars Science Lab |
| SWaT | Industrial control | iTrust Secure Water Treatment |
| WADI | Industrial control | iTrust Water Distribution |
| SMD | Server metrics | Server Machine Dataset |

Download all public datasets:

```bash
bash scripts/download_all.sh
```

**SWaT and WADI** require registration with iTrust (https://itrust.sutd.edu.sg/). If you do not have access, dummy data generators are provided:

```bash
python scripts/generate_dummy_swat.py
python scripts/generate_dummy_wadi.py
```

Results from dummy SWaT/WADI data are automatically marked `[DUMMY]` in output tables.

## Quick Start

```bash
python -m vector --dataset nab --mode all
```

This runs the full pipeline for NAB: preprocesses raw data, runs a 1500-trial NSGA-II search, evaluates 3 baselines (default, grid search, random search), prints results tables, and generates Pareto front plots.

## CLI Reference

```
python -m vector --dataset <name> --mode <stage> [--config <path>] [--jobs <n>]
```

**Arguments:**

- `--dataset`: Dataset to process. Choices: `nab`, `ucr`, `mba`, `smap`, `msl`, `swat`, `wadi`, `smd`, `all`
- `--mode`: Pipeline stage to run. Choices: `preprocess`, `search`, `baseline`, `eval`, `plot`, `all`
- `--config`: Path to search config YAML (default: `experiments/configs/search.yaml`)
- `--jobs`: Override parallel worker count for search

**Examples:**

```bash
# Preprocess a single dataset
python -m vector --dataset nab --mode preprocess

# Run search with custom config and 8 workers
python -m vector --dataset ucr --mode search --config my_config.yaml --jobs 8

# Evaluate baselines for all datasets
python -m vector --dataset all --mode baseline

# Print results tables
python -m vector --dataset all --mode eval

# Generate Pareto front plots
python -m vector --dataset smap --mode plot
```

## Reproduction Instructions

To reproduce the full results:

1. **Download or generate data** for all 8 datasets (see Dataset Download above).

2. **Preprocess all datasets:**
   ```bash
   python -m vector --dataset all --mode preprocess
   ```

3. **Run NSGA-II search** for each dataset (1500 trials per dataset; this takes time):
   ```bash
   python -m vector --dataset all --mode search
   ```
   If interrupted, search resumes automatically from the SQLite journal.

4. **Run baselines** (default, grid search, random search):
   ```bash
   python -m vector --dataset all --mode baseline
   ```

5. **Print results tables:**
   ```bash
   python -m vector --dataset all --mode eval
   ```
   This prints Table 3 (Precision/Recall/F1) and Table 4 (F1/Time/RACS).

6. **Generate Pareto front plots:**
   ```bash
   python -m vector --dataset all --mode plot
   ```

Or run everything in one command:

```bash
python -m vector --dataset all --mode all
```

## Results Tables

Running `--mode eval` prints two tables matching the TransNAS-TSAD layout:

- **Table 3** -- Detection performance: Precision / Recall / F1 per method per dataset
- **Table 4** -- Efficiency: F1 / Training Time / RACS per method per dataset

Results from dummy SWaT/WADI data are marked `[DUMMY]` in column headers.

## Project Structure

```
vector/              Core library
  data/              Data loading, preprocessing, pipeline
  esn/               Echo State Network implementation
  scoring/           MDRS scorer and SPOT thresholding
  evaluation/        Metrics (point-adjust F1, AUROC, RACS)
  search/            NSGA-II search engine and objective
  baselines.py       Default, grid search, random search baselines
  pareto.py          Pareto front extraction and plotting
  results.py         Results table formatting
  __main__.py        CLI entry point

experiments/
  configs/           search.yaml, datasets.yaml
  results/           Per-dataset JSON results and plots

data/                Raw and processed dataset storage
scripts/             Download and dummy data generation scripts
tests/               Test suite
```

## Testing

Run all tests:

```bash
pytest
```

Run with coverage threshold:

```bash
pytest --cov=vector --cov-fail-under=80
```
