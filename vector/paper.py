"""Paper artifact generation: LaTeX tables, statistical tests, and plots.

Produces publication-ready LaTeX tables (Table 3: P/R/F1, Table 4: F1/Time/RACS),
Wilcoxon signed-rank statistical significance tests, convergence plots, Pareto
front evolution plots, and ablation study plots for VECTOR vs baselines.
All output written to experiments/paper/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from vector.results import ALL_DATASETS, ALL_METHODS, collect_results, is_dummy_data  # noqa: E402


def save_figure(fig: plt.Figure, output_dir: str | Path, name: str) -> None:
    """Save figure as both PNG (300 DPI) and PDF (TrueType fonts).

    Creates output_dir if it does not exist. Closes the figure after saving.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _fmt3(value: object) -> str:
    """Format a numeric value to 3 decimal places, or return '--'."""
    if value == "N/A" or value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _escape_latex(text: str) -> str:
    """Escape underscores for LaTeX."""
    return text.replace("_", r"\_")


def generate_latex_table3(
    results: dict[str, dict[str, dict]],
    dummy_datasets: set[str],
    output_dir: str | Path,
) -> str:
    """Generate LaTeX Table 3: Detection Performance (P/R/F1 per method per dataset).

    Bolds the best F1 per dataset column. Marks dummy datasets with a dagger
    symbol in the header. Writes to {output_dir}/table3.tex.

    Returns the LaTeX string.
    """
    datasets = ALL_DATASETS

    # Build header
    col_spec = "l" + "c" * len(datasets)
    header_cells = ["Method"]
    for ds in datasets:
        label = ds + r"$^\dagger$" if ds in dummy_datasets else ds
        header_cells.append(label)

    # Find best F1 per dataset
    best_f1: dict[str, tuple[float, str]] = {}
    for ds in datasets:
        best_val = -1.0
        best_method = ""
        for method in ALL_METHODS:
            entry = results.get(ds, {}).get(method, {})
            f1 = entry.get("f1", "N/A")
            if f1 != "N/A" and f1 is not None:
                try:
                    fv = float(f1)
                    if fv > best_val:
                        best_val = fv
                        best_method = method
                except (TypeError, ValueError):
                    pass
        if best_method:
            best_f1[ds] = (best_val, best_method)

    # Build rows
    row_lines = []
    for method in ALL_METHODS:
        cells = [_escape_latex(method)]
        for ds in datasets:
            entry = results.get(ds, {}).get(method, {})
            p = _fmt3(entry.get("precision", "N/A"))
            r = _fmt3(entry.get("recall", "N/A"))
            f1 = _fmt3(entry.get("f1", "N/A"))

            if p == "--" and r == "--" and f1 == "--":
                cells.append("--")
            else:
                cell = f"{p}/{r}/{f1}"
                if ds in best_f1 and best_f1[ds][1] == method:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        row_lines.append(" & ".join(cells) + r" \\")

    # Assemble LaTeX
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join(header_cells) + r" \\",
        r"\hline",
    ]
    lines.extend(row_lines)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    latex = "\n".join(lines)

    out_path = Path(output_dir) / "table3.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    return latex


def generate_latex_table4(
    results: dict[str, dict[str, dict]],
    dummy_datasets: set[str],
    output_dir: str | Path,
) -> str:
    """Generate LaTeX Table 4: Efficiency (F1/Time/RACS per method per dataset).

    Bolds the best RACS per dataset column. Marks dummy datasets with a dagger
    symbol in the header. Writes to {output_dir}/table4.tex.

    Returns the LaTeX string.
    """
    datasets = ALL_DATASETS

    col_spec = "l" + "c" * len(datasets)
    header_cells = ["Method"]
    for ds in datasets:
        label = ds + r"$^\dagger$" if ds in dummy_datasets else ds
        header_cells.append(label)

    # Find best RACS per dataset
    best_racs: dict[str, tuple[float, str]] = {}
    for ds in datasets:
        best_val = -1.0
        best_method = ""
        for method in ALL_METHODS:
            entry = results.get(ds, {}).get(method, {})
            racs_val = entry.get("racs", "N/A")
            if racs_val != "N/A" and racs_val is not None:
                try:
                    rv = float(racs_val)
                    if rv > best_val:
                        best_val = rv
                        best_method = method
                except (TypeError, ValueError):
                    pass
        if best_method:
            best_racs[ds] = (best_val, best_method)

    row_lines = []
    for method in ALL_METHODS:
        cells = [_escape_latex(method)]
        for ds in datasets:
            entry = results.get(ds, {}).get(method, {})
            f1 = _fmt3(entry.get("f1", "N/A"))
            time_val = entry.get("training_time", "N/A")
            racs_val = _fmt3(entry.get("racs", "N/A"))

            if time_val != "N/A" and time_val is not None:
                try:
                    time_str = f"{float(time_val):.1f}"
                except (TypeError, ValueError):
                    time_str = "--"
            else:
                time_str = "--"

            if f1 == "--" and time_str == "--" and racs_val == "--":
                cells.append("--")
            else:
                cell = f"{f1}/{time_str}/{racs_val}"
                if ds in best_racs and best_racs[ds][1] == method:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        row_lines.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join(header_cells) + r" \\",
        r"\hline",
    ]
    lines.extend(row_lines)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    latex = "\n".join(lines)

    out_path = Path(output_dir) / "table4.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    return latex


def significance_tests(
    results: dict[str, dict[str, dict]],
    reference_method: str = "VECTOR",
) -> list[dict]:
    """Run Wilcoxon signed-rank test comparing reference vs each baseline.

    Requires >= 6 paired non-zero differences. Returns list of dicts with
    baseline, p_value, statistic, n_pairs, and optional note.
    """
    baselines = [m for m in ALL_METHODS if m != reference_method]
    tests = []

    for baseline in baselines:
        ref_f1s: list[float] = []
        base_f1s: list[float] = []

        for ds in ALL_DATASETS:
            ref = results.get(ds, {}).get(reference_method, {}).get("f1")
            base = results.get(ds, {}).get(baseline, {}).get("f1")
            if ref not in (None, "N/A") and base not in (None, "N/A"):
                try:
                    ref_f1s.append(float(ref))
                    base_f1s.append(float(base))
                except (TypeError, ValueError):
                    pass

        if len(ref_f1s) < 6:
            tests.append({
                "baseline": baseline,
                "p_value": None,
                "statistic": None,
                "n_pairs": len(ref_f1s),
                "note": f"Insufficient pairs for Wilcoxon test (need >= 6, got {len(ref_f1s)})",
            })
            continue

        diffs = [r - b for r, b in zip(ref_f1s, base_f1s)]
        if all(d == 0.0 for d in diffs):
            tests.append({
                "baseline": baseline,
                "p_value": 1.0,
                "statistic": 0.0,
                "n_pairs": len(ref_f1s),
                "note": "All differences are zero",
            })
            continue

        stat, p_value = wilcoxon(ref_f1s, base_f1s, alternative="two-sided")
        tests.append({
            "baseline": baseline,
            "p_value": float(p_value),
            "statistic": float(stat),
            "n_pairs": len(ref_f1s),
        })

    return tests


def generate_significance_table(
    test_results: list[dict],
    output_dir: str | Path,
) -> str:
    """Format significance test results as a LaTeX table.

    Columns: Baseline, N pairs, Statistic, p-value, Significant (alpha=0.05).
    Writes to {output_dir}/significance.tex.

    Returns the LaTeX string.
    """
    header_cells = [
        "Baseline",
        "N pairs",
        "Statistic",
        "p-value",
        r"Significant ($\alpha=0.05$)",
    ]

    row_lines = []
    for entry in test_results:
        baseline = _escape_latex(entry["baseline"])
        n_pairs = str(entry["n_pairs"])

        if entry.get("note") and entry["p_value"] is None:
            stat_str = "--"
            p_str = "--"
            sig_str = entry["note"]
        else:
            p_val = entry["p_value"]
            stat_val = entry.get("statistic")
            stat_str = f"{stat_val:.1f}" if stat_val is not None else "--"
            p_str = f"{p_val:.4f}" if p_val is not None else "--"

            if p_val is not None and p_val < 0.05:
                sig_str = "Yes"
            elif p_val is not None:
                sig_str = "No"
            else:
                sig_str = "--"

            if entry.get("note"):
                sig_str += f" ({entry['note']})"

        cells = [baseline, n_pairs, stat_str, p_str, sig_str]
        row_lines.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        " & ".join(header_cells) + r" \\",
        r"\hline",
    ]
    lines.extend(row_lines)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    latex = "\n".join(lines)

    out_path = Path(output_dir) / "significance.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")

    return latex


def generate_tables_and_stats(
    datasets: list[str],
    dataset_config: dict,
    output_dir: str = "experiments/paper",
) -> None:
    """Orchestrate table and significance test generation.

    1. Collects results via collect_results().
    2. Determines dummy datasets via is_dummy_data().
    3. Generates LaTeX Table 3 (P/R/F1).
    4. Generates LaTeX Table 4 (F1/Time/RACS).
    5. Runs Wilcoxon significance tests.
    6. Generates significance results table.
    7. Prints summary of generated files.
    """
    results = collect_results(datasets)

    dummy_datasets = {
        ds for ds in datasets if is_dummy_data(ds, dataset_config)
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generate_latex_table3(results, dummy_datasets, output_dir)
    print(f"  Table 3 (P/R/F1) -> {out / 'table3.tex'}")

    generate_latex_table4(results, dummy_datasets, output_dir)
    print(f"  Table 4 (F1/Time/RACS) -> {out / 'table4.tex'}")

    test_results = significance_tests(results)
    generate_significance_table(test_results, output_dir)
    print(f"  Significance tests -> {out / 'significance.tex'}")

    print(f"\nGenerated {3} LaTeX tables in {output_dir}/")
    for entry in test_results:
        p = entry["p_value"]
        note = entry.get("note", "")
        p_str = f"p={p:.4f}" if p is not None else note
        print(f"  {entry['baseline']}: {p_str} (n={entry['n_pairs']})")


def plot_convergence(
    dataset_name: str,
    search_config: dict,
    output_dir: str | Path,
) -> str | None:
    """Plot convergence curve: running-best F1 vs trial number.

    Shows gray scatter dots for per-trial F1 (alpha=0.2) and a solid blue
    line for the running best. Saves dual PNG+PDF via save_figure.

    Returns the output path stem or None if the study cannot be loaded.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from vector.search.engine import create_or_load_study

        study = create_or_load_study(dataset_name, search_config)
    except Exception as exc:
        print(f"Warning: cannot load study for {dataset_name}: {exc}")
        return None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"Warning: no completed trials for {dataset_name}")
        return None

    completed.sort(key=lambda t: t.number)
    f1_values = np.array([1.0 - t.values[0] for t in completed])
    trial_numbers = np.array([t.number for t in completed])
    running_best = np.maximum.accumulate(f1_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(trial_numbers, f1_values, color="gray", alpha=0.2, s=8, label="Per-trial F1")
    ax.plot(trial_numbers, running_best, color="blue", linewidth=2, label="Running best")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Convergence - {dataset_name}")
    ax.legend()

    name = f"convergence_{dataset_name}"
    save_figure(fig, output_dir, name)
    return name


def plot_pareto_evolution(
    dataset_name: str,
    search_config: dict,
    output_dir: str | Path,
    batches: list[int] | None = None,
) -> str | None:
    """Plot Pareto front evolution at different trial batch sizes.

    Overlays non-dominated fronts at each batch checkpoint (default:
    100, 500, 1000, 1500 trials). Uses viridis colormap for progression.
    Saves dual PNG+PDF via save_figure.

    Returns the output path stem or None if the study cannot be loaded.
    """
    if batches is None:
        batches = [100, 500, 1000, 1500]

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from vector.search.engine import create_or_load_study

        study = create_or_load_study(dataset_name, search_config)
    except Exception as exc:
        print(f"Warning: cannot load study for {dataset_name}: {exc}")
        return None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"Warning: no completed trials for {dataset_name}")
        return None

    completed.sort(key=lambda t: t.number)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(batches)))

    for batch_size, color in zip(batches, colors):
        subset = [t for t in completed if t.number < batch_size]
        if not subset:
            continue

        # Extract objectives: (effective_size, 1-F1) -- both minimized
        points = np.array([[t.values[1], t.values[0]] for t in subset])

        # Find non-dominated front via pairwise dominance
        front_mask = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if not front_mask[i]:
                continue
            for j in range(len(points)):
                if i == j or not front_mask[j]:
                    continue
                # j dominates i if j <= i on both and strictly < on at least one
                if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                        (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                    front_mask[i] = False
                    break

        front = points[front_mask]
        # Convert back to (effective_size, F1) for display
        ax.scatter(
            front[:, 0],
            1.0 - front[:, 1],
            color=color,
            s=30,
            label=f"After {batch_size} trials",
            alpha=0.8,
        )

    ax.set_xlabel("Effective Reservoir Size (n_res / k)")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Pareto Front Evolution - {dataset_name}")
    ax.legend()

    name = f"pareto_evolution_{dataset_name}"
    save_figure(fig, output_dir, name)
    return name


def plot_f1_breakdown(
    results: dict[str, dict[str, dict]],
    dummy_datasets: set[str],
    output_dir: str | Path,
) -> None:
    """Grouped bar chart: datasets on x-axis, one bar per method, F1 on y-axis.

    Marks dummy datasets with "(D)" suffix. Handles N/A values by plotting
    zero-height bars with hatching. Legend outside plot area.
    """
    datasets = ALL_DATASETS
    methods = ALL_METHODS
    n_datasets = len(datasets)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_methods
    colors = [plt.cm.tab10(i) for i in range(n_methods)]

    for m_idx, method in enumerate(methods):
        x_positions = np.arange(n_datasets) + m_idx * bar_width
        heights = []
        hatch_flags = []

        for ds in datasets:
            entry = results.get(ds, {}).get(method, {})
            f1 = entry.get("f1", "N/A")
            if f1 == "N/A" or f1 is None:
                heights.append(0.0)
                hatch_flags.append(True)
            else:
                try:
                    heights.append(float(f1))
                    hatch_flags.append(False)
                except (TypeError, ValueError):
                    heights.append(0.0)
                    hatch_flags.append(True)

        bars = ax.bar(
            x_positions, heights, bar_width,
            label=method, color=colors[m_idx], edgecolor="black", linewidth=0.5,
        )
        for bar, is_na in zip(bars, hatch_flags):
            if is_na:
                bar.set_hatch("//")
                bar.set_alpha(0.3)

    x_labels = [
        f"{ds} (D)" if ds in dummy_datasets else ds
        for ds in datasets
    ]
    ax.set_xticks(np.arange(n_datasets) + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Breakdown by Dataset and Method")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(fig, output_dir, "f1_breakdown")


def plot_racs_scatter(
    results: dict[str, dict[str, dict]],
    output_dir: str | Path,
) -> None:
    """Scatter plot: effective_size on x-axis, RACS on y-axis, colored by method.

    Each point represents one dataset for one method. Skips N/A entries.
    """
    methods = ALL_METHODS
    markers = ["o", "s", "^", "D"]
    colors = [plt.cm.tab10(i) for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for m_idx, method in enumerate(methods):
        xs = []
        ys = []
        for ds in ALL_DATASETS:
            entry = results.get(ds, {}).get(method, {})
            eff_size = entry.get("effective_size", "N/A")
            racs = entry.get("racs", "N/A")
            if eff_size in ("N/A", None) or racs in ("N/A", None):
                continue
            try:
                xs.append(float(eff_size))
                ys.append(float(racs))
            except (TypeError, ValueError):
                continue

        if xs:
            ax.scatter(
                xs, ys,
                color=colors[m_idx],
                marker=markers[m_idx],
                s=60,
                label=method,
                alpha=0.8,
            )

    ax.set_xlabel("Effective Reservoir Size")
    ax.set_ylabel("RACS")
    ax.set_title("RACS vs Effective Size by Method")
    ax.legend()

    save_figure(fig, output_dir, "racs_scatter")


def generate_all_artifacts(
    datasets: list[str],
    dataset_config: dict,
    search_config: dict,
    output_dir: str = "experiments/paper",
) -> None:
    """Orchestrate generation of all paper artifacts.

    Runs table/stats generation, F1 breakdown, RACS scatter, convergence,
    Pareto evolution, and ablation plots for each dataset.
    """
    print("Generating paper artifacts...")

    # 1. Tables and significance tests
    generate_tables_and_stats(datasets, dataset_config, output_dir)

    # 2. Collect results for plots
    results = collect_results(datasets)
    dummy_datasets = {
        ds for ds in datasets if is_dummy_data(ds, dataset_config)
    }

    # 3. F1 breakdown bar chart
    plot_f1_breakdown(results, dummy_datasets, output_dir)
    print(f"  F1 breakdown -> {output_dir}/f1_breakdown.png/.pdf")

    # 4. RACS scatter plot
    plot_racs_scatter(results, output_dir)
    print(f"  RACS scatter -> {output_dir}/racs_scatter.png/.pdf")

    # 5. Per-dataset plots
    convergence_count = 0
    pareto_count = 0
    ablation_count = 0
    for ds in datasets:
        result = plot_convergence(ds, search_config, output_dir)
        if result:
            convergence_count += 1

        result = plot_pareto_evolution(ds, search_config, output_dir)
        if result:
            pareto_count += 1

        result = plot_ablation(ds, search_config, output_dir)
        if result:
            ablation_count += 1

    total = 3 + convergence_count + pareto_count + ablation_count  # 3 tables
    print(f"\nTotal artifacts generated: {total}")
    print(f"  Tables: 3, F1 breakdown: 1, RACS scatter: 1")
    print(f"  Convergence: {convergence_count}, Pareto: {pareto_count}, Ablation: {ablation_count}")


_ABLATION_PARAMS = ["n_res", "rho", "sigma", "sparsity", "alpha", "k", "n_wash"]


def plot_ablation(
    dataset_name: str,
    search_config: dict,
    output_dir: str | Path,
) -> str | None:
    """Plot ablation study: F1 sensitivity to each of the 7 ESN hyperparameters.

    Mines existing Optuna trial history rather than running new experiments.
    For each parameter, bins trials across the search range and plots mean F1
    with shaded standard deviation. Produces a 2x4 subplot grid (7 params +
    1 hidden). Saves dual PNG+PDF via save_figure.

    Returns the output path stem or None if the study cannot be loaded.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from vector.search.engine import create_or_load_study

        study = create_or_load_study(dataset_name, search_config)
    except Exception as exc:
        print(f"Warning: cannot load study for {dataset_name}: {exc}")
        return None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"Warning: no completed trials for {dataset_name}")
        return None

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.flatten()

    for idx, param_name in enumerate(_ABLATION_PARAMS):
        ax = axes_flat[idx]

        # Extract (param_value, f1) pairs from completed trials
        param_vals = []
        f1_vals = []
        for trial in completed:
            if param_name in trial.params:
                param_vals.append(float(trial.params[param_name]))
                f1_vals.append(1.0 - trial.values[0])

        if len(param_vals) < 2:
            ax.set_title(param_name)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        param_arr = np.array(param_vals)
        f1_arr = np.array(f1_vals)

        # Bin trials and compute mean/std F1 per bin
        bin_edges = np.histogram_bin_edges(param_arr, bins=15)
        bin_centers = []
        bin_means = []
        bin_stds = []

        for b in range(len(bin_edges) - 1):
            mask = (param_arr >= bin_edges[b]) & (param_arr < bin_edges[b + 1])
            # Include right edge in last bin
            if b == len(bin_edges) - 2:
                mask = (param_arr >= bin_edges[b]) & (param_arr <= bin_edges[b + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_means.append(f1_arr[mask].mean())
                bin_stds.append(f1_arr[mask].std())

        centers = np.array(bin_centers)
        means = np.array(bin_means)
        stds = np.array(bin_stds)

        ax.plot(centers, means, color="blue", linewidth=1.5)
        ax.fill_between(centers, means - stds, means + stds, alpha=0.2, color="blue")
        ax.set_xlabel(param_name)
        ax.set_ylabel("F1 Score")
        ax.set_title(param_name)

    # Hide the 8th (empty) subplot
    axes_flat[7].set_visible(False)

    fig.suptitle(f"Ablation Study - {dataset_name}", fontsize=14)
    fig.tight_layout()

    name = f"ablation_{dataset_name}"
    save_figure(fig, output_dir, name)
    return name
