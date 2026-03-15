"""Pareto front extraction, RACS ranking, JSON serialization, and scatter plotting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vector.evaluation.metrics import racs

logger = logging.getLogger(__name__)


def _to_serializable(obj: object) -> object:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def extract_pareto(study) -> list[dict]:
    """Extract Pareto front trials with RACS ranking.

    Uses Optuna's built-in ``study.best_trials`` for Pareto extraction
    (never hand-rolled dominance). Each trial is scored with RACS and
    the list is returned sorted by RACS descending with 1-indexed rank.

    Parameters
    ----------
    study : optuna.study.Study
        Completed multi-objective study with directions
        ``["minimize", "minimize"]`` where objective 0 is ``1 - F1``
        and objective 1 is ``n_res / k``.

    Returns
    -------
    list[dict]
        Pareto-optimal trials ranked by RACS descending. Each dict has
        keys: trial_number, params, f1, effective_size, racs, rank.
    """
    pareto_trials = study.best_trials
    if not pareto_trials:
        return []

    results = []
    for trial in pareto_trials:
        f1 = 1.0 - trial.values[0]
        n_res = trial.params["n_res"]
        k = trial.params["k"]
        effective_size = trial.values[1]
        racs_score = racs(f1, n_res, k)
        results.append(
            {
                "trial_number": trial.number,
                "params": dict(trial.params),
                "f1": f1,
                "effective_size": effective_size,
                "racs": racs_score,
            }
        )

    results.sort(key=lambda r: r["racs"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return results


def save_pareto_results(
    pareto_results: list[dict],
    dataset_name: str,
    output_dir: str | Path,
) -> str:
    """Save Pareto front results to JSON.

    Parameters
    ----------
    pareto_results : list[dict]
        Output from :func:`extract_pareto`.
    dataset_name : str
        Dataset identifier (e.g. ``"NAB"``).
    output_dir : str or Path
        Base output directory. Results are written to
        ``{output_dir}/{dataset_name}/pareto.json``.

    Returns
    -------
    str
        Path to the written JSON file.
    """
    output = {
        "dataset": dataset_name,
        "n_pareto_trials": len(pareto_results),
        "best_racs_trial": (
            pareto_results[0]["trial_number"] if pareto_results else None
        ),
        "trials": pareto_results,
    }
    output = _to_serializable(output)

    out_path = Path(output_dir) / dataset_name / "pareto.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return str(out_path)


def plot_pareto(
    pareto_results: list[dict],
    dataset_name: str,
    output_dir: str | Path,
) -> str | None:
    """Create Pareto scatter plot with RACS coloring.

    Plots effective reservoir size vs F1 with points colored by RACS
    and a red star marker on the best-RACS trial.

    Parameters
    ----------
    pareto_results : list[dict]
        Output from :func:`extract_pareto`.
    dataset_name : str
        Dataset identifier used in title and filename.
    output_dir : str or Path
        Directory for plot output. Plot saved to
        ``{output_dir}/pareto_{dataset_name}.png``.

    Returns
    -------
    str or None
        Path to saved PNG, or ``None`` if *pareto_results* is empty.
    """
    if not pareto_results:
        logger.warning("Empty Pareto results for %s -- skipping plot.", dataset_name)
        return None

    eff_sizes = [r["effective_size"] for r in pareto_results]
    f1s = [r["f1"] for r in pareto_results]
    racs_scores = [r["racs"] for r in pareto_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        eff_sizes,
        f1s,
        c=racs_scores,
        cmap="viridis",
        s=60,
        edgecolors="k",
        linewidths=0.5,
        zorder=2,
    )
    plt.colorbar(scatter, ax=ax, label="RACS")

    best = pareto_results[0]
    ax.scatter(
        [best["effective_size"]],
        [best["f1"]],
        marker="*",
        s=300,
        c="red",
        edgecolors="k",
        linewidths=1,
        zorder=3,
        label=f"Best RACS={best['racs']:.3f}",
    )

    ax.set_xlabel("Effective Reservoir Size (n_res / k)")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Pareto Front - {dataset_name}")
    ax.legend()

    output_path = Path(output_dir) / f"pareto_{dataset_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)
