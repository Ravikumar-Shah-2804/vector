"""Results table formatters: Table 3 (P/R/F1) and Table 4 (F1/Time/RACS).

Produces publication-ready ASCII tables matching the TransNAS-TSAD layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from tabulate import tabulate

ALL_DATASETS = ["NAB", "UCR", "MBA", "SMAP", "MSL", "SWaT", "WADI", "SMD"]
ALL_METHODS = ["Default", "Grid Search", "Random Search", "VECTOR"]

_BASELINE_KEY_MAP = {
    "default": "Default",
    "grid_search": "Grid Search",
    "random_search": "Random Search",
}


def is_dummy_data(dataset_name: str, dataset_config: dict) -> bool:
    """Check whether a dataset is using dummy/synthetic data.

    Only SWaT and WADI can be dummy (they require iTrust registration).
    Returns True if the real data file is absent.
    """
    if dataset_name not in {"SWaT", "WADI"}:
        return False

    ds_cfg = dataset_config.get("datasets", {}).get(dataset_name, {})
    raw_path = Path(ds_cfg.get("raw_path", ""))

    real_files = {
        "SWaT": "SWaT_Dataset_Normal_v1.xlsx",
        "WADI": "WADI_14days.csv",
    }

    real_file = raw_path / real_files[dataset_name]
    return not real_file.exists()


def collect_results(
    datasets: list[str],
    results_dir: str = "experiments/results",
) -> dict[str, dict[str, dict]]:
    """Collect results from baseline.json and pareto.json for each dataset.

    Returns nested dict: {dataset: {method: {f1, precision, recall, racs, training_time}}}.
    Missing data is filled with "N/A" strings.
    """
    results: dict[str, dict[str, dict]] = {}

    for ds in datasets:
        ds_results: dict[str, dict] = {}
        ds_dir = Path(results_dir) / ds

        # Load baseline results
        baseline_path = ds_dir / "baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_data = json.load(f)

            for raw_key, display_name in _BASELINE_KEY_MAP.items():
                entry = baseline_data.get(raw_key, {})
                if not entry:
                    ds_results[display_name] = _na_entry()
                    continue

                # Default baseline has direct f1 key; grid/random have best_f1
                f1 = entry.get("f1", entry.get("best_f1", "N/A"))

                ds_results[display_name] = {
                    "f1": f1,
                    "precision": entry.get("precision", "N/A"),
                    "recall": entry.get("recall", "N/A"),
                    "racs": entry.get("racs", "N/A"),
                    "training_time": entry.get("training_time", "N/A"),
                    "effective_size": entry.get(
                        "effective_size",
                        entry.get("best_params", {}).get("n_res", "N/A"),
                    ),
                }
        else:
            for name in ["Default", "Grid Search", "Random Search"]:
                ds_results[name] = _na_entry()

        # Load VECTOR (Pareto) results
        pareto_path = ds_dir / "pareto.json"
        if pareto_path.exists():
            with open(pareto_path) as f:
                pareto_data = json.load(f)

            trials = pareto_data.get("trials", [])
            if trials:
                best = trials[0]  # First entry is best-RACS
                ds_results["VECTOR"] = {
                    "f1": best.get("f1", "N/A"),
                    "precision": best.get("precision", "N/A"),
                    "recall": best.get("recall", "N/A"),
                    "racs": best.get("racs", "N/A"),
                    "training_time": best.get("training_time", "N/A"),
                    "effective_size": best.get("effective_size", "N/A"),
                }
            else:
                ds_results["VECTOR"] = _na_entry()
        else:
            ds_results["VECTOR"] = _na_entry()

        results[ds] = ds_results

    return results


def _na_entry() -> dict:
    """Return a result entry with all N/A values."""
    return {
        "f1": "N/A",
        "precision": "N/A",
        "recall": "N/A",
        "racs": "N/A",
        "training_time": "N/A",
        "effective_size": "N/A",
    }


def _fmt(value: object, places: int = 4) -> str:
    """Format a numeric value or return N/A."""
    if value == "N/A" or value is None:
        return "N/A"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_table3(
    results_by_dataset: dict[str, dict[str, dict]],
    dummy_datasets: set[str],
) -> str:
    """Format Table 3: Detection Performance (Precision / Recall / F1).

    Returns an ASCII grid table string.
    """
    headers = ["Method"]
    for ds in ALL_DATASETS:
        label = f"{ds} [DUMMY]" if ds in dummy_datasets else ds
        headers.append(label)

    rows = []
    for method in ALL_METHODS:
        row = [method]
        for ds in ALL_DATASETS:
            entry = results_by_dataset.get(ds, {}).get(method, _na_entry())
            p = _fmt(entry.get("precision"))
            r = _fmt(entry.get("recall"))
            f1 = _fmt(entry.get("f1"))
            row.append(f"{p} / {r} / {f1}")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def format_table4(
    results_by_dataset: dict[str, dict[str, dict]],
    dummy_datasets: set[str],
) -> str:
    """Format Table 4: Efficiency (F1 / Training Time / RACS).

    Returns an ASCII grid table string.
    """
    headers = ["Method"]
    for ds in ALL_DATASETS:
        label = f"{ds} [DUMMY]" if ds in dummy_datasets else ds
        headers.append(label)

    rows = []
    for method in ALL_METHODS:
        row = [method]
        for ds in ALL_DATASETS:
            entry = results_by_dataset.get(ds, {}).get(method, _na_entry())
            f1 = _fmt(entry.get("f1"))
            time_s = _fmt(entry.get("training_time"), places=1)
            r = _fmt(entry.get("racs"))
            row.append(f"{f1} / {time_s} / {r}")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def print_results(
    datasets: list[str],
    dataset_config: dict,
    results_dir: str = "experiments/results",
) -> None:
    """Collect and print Tables 3 and 4 to stdout."""
    results = collect_results(datasets, results_dir)

    dummy_datasets = {
        ds for ds in datasets if is_dummy_data(ds, dataset_config)
    }

    print("\nTable 3: Detection Performance (P/R/F1)")
    print("=" * 60)
    print(format_table3(results, dummy_datasets))

    print("\nTable 4: Efficiency (F1/Time/RACS)")
    print("=" * 60)
    print(format_table4(results, dummy_datasets))
