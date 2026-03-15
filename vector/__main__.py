"""CLI entry point for the VECTOR pipeline.

Usage: python -m vector --dataset nab --mode all
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import Optional

DATASET_MAP: dict[str, str] = {
    "nab": "NAB",
    "ucr": "UCR",
    "mba": "MBA",
    "smap": "SMAP",
    "msl": "MSL",
    "swat": "SWaT",
    "wadi": "WADI",
    "smd": "SMD",
}

MODE_ORDER = ["preprocess", "search", "baseline", "eval", "plot", "paper"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vector",
        description="VECTOR: Multi-objective NAS for Reservoir Computing anomaly detection",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_MAP.keys()) + ["all"],
        help="Dataset to process (or 'all' for every dataset)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_ORDER + ["all"],
        help="Pipeline stage to run (or 'all' for full pipeline)",
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/search.yaml",
        help="Path to search config YAML (default: experiments/configs/search.yaml)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Override n_jobs for parallel workers",
    )
    return parser


def resolve_datasets(dataset_arg: str) -> list[str]:
    """Resolve CLI dataset argument to list of config-cased dataset names."""
    if dataset_arg == "all":
        return list(DATASET_MAP.values())
    return [DATASET_MAP[dataset_arg]]


def _load_sequences(
    dataset_name: str, dataset_config: dict
) -> Optional[list[dict]]:
    """Load preprocessed .npy sequences from processed directory.

    Returns list of dicts with train/val/test/labels arrays,
    or None if directory does not exist.
    """
    ds_cfg = dataset_config["datasets"][dataset_name]
    processed_path = Path(ds_cfg["processed_path"])

    if not processed_path.exists():
        print(
            f"ERROR: Processed data not found for {dataset_name} at "
            f"{processed_path}. Run --mode preprocess first."
        )
        return None

    sequences = []
    subdirs = sorted(
        d for d in processed_path.iterdir() if d.is_dir()
    )

    if not subdirs:
        print(f"ERROR: No sequence subdirectories in {processed_path}")
        return None

    import numpy as np

    for subdir in subdirs:
        seq = {
            "name": subdir.name,
            "train": np.load(subdir / "train.npy"),
            "val": np.load(subdir / "val.npy"),
            "test": np.load(subdir / "test.npy"),
            "labels": np.load(subdir / "test_labels.npy"),
        }
        sequences.append(seq)

    return sequences


def _run_preprocess(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Run preprocessing for each dataset."""
    from vector.data.pipeline import preprocess_dataset

    for ds in datasets:
        print(f"[{ds}] Preprocessing...")
        result = preprocess_dataset(ds, config=dataset_config)
        if result is not None:
            print(f"[{ds}] Done: {result['n_sequences']} sequences")
        else:
            print(f"[{ds}] Skipped (no loader or missing files)")


def _run_search(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Run NSGA-II search for each dataset."""
    from vector.search.engine import run_search

    for ds in datasets:
        print(f"[{ds}] Loading sequences...")
        sequences = _load_sequences(ds, dataset_config)
        if sequences is None:
            continue
        print(f"[{ds}] Starting search ({len(sequences)} sequences)...")
        run_search(ds, sequences, search_config, dataset_config)


def _run_baseline(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Run baseline evaluations for each dataset."""
    from vector.baselines import run_all_baselines

    for ds in datasets:
        print(f"[{ds}] Loading sequences...")
        sequences = _load_sequences(ds, dataset_config)
        if sequences is None:
            continue
        print(f"[{ds}] Running baselines...")
        run_all_baselines(sequences, ds, search_config, dataset_config)


def _run_eval(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Collect results and print Tables 3 and 4."""
    from vector.results import is_dummy_data, print_results

    print_results(datasets, dataset_config)


def _run_plot(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Generate Pareto front plots for each dataset."""
    from vector.pareto import extract_pareto, plot_pareto
    from vector.search.engine import create_or_load_study

    for ds in datasets:
        print(f"[{ds}] Loading study...")
        try:
            study = create_or_load_study(ds, search_config)
        except Exception as e:
            print(f"[{ds}] Could not load study: {e}")
            continue

        pareto = extract_pareto(study)
        if not pareto:
            print(f"[{ds}] No Pareto solutions found")
            continue

        output_dir = os.path.join("experiments", "results", ds)
        plot_pareto(pareto, ds, output_dir)
        print(f"[{ds}] Pareto plot saved")


def _run_paper(
    datasets: list[str], dataset_config: dict, search_config: dict
) -> None:
    """Generate all paper artifacts (tables, stats, plots)."""
    from vector.paper import generate_all_artifacts

    generate_all_artifacts(datasets, dataset_config, search_config)


MODE_DISPATCH = {
    "preprocess": _run_preprocess,
    "search": _run_search,
    "baseline": _run_baseline,
    "eval": _run_eval,
    "plot": _run_plot,
    "paper": _run_paper,
}


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the VECTOR CLI."""
    from vector.data.config import load_config
    from vector.search.config import load_search_config

    args = build_parser().parse_args(argv)

    dataset_config = load_config()
    search_config = load_search_config(args.config)

    if args.jobs is not None:
        search_config["optimization"]["n_jobs"] = args.jobs

    datasets = resolve_datasets(args.dataset)

    if args.mode == "all":
        for mode in MODE_ORDER:
            print(f"\n{'='*60}")
            print(f"  Stage: {mode}")
            print(f"{'='*60}")
            MODE_DISPATCH[mode](datasets, dataset_config, search_config)
    else:
        MODE_DISPATCH[args.mode](datasets, dataset_config, search_config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
