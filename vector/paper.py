"""Paper artifact generation: LaTeX tables and statistical significance tests.

Produces publication-ready LaTeX tables (Table 3: P/R/F1, Table 4: F1/Time/RACS)
and Wilcoxon signed-rank statistical significance tests for VECTOR vs baselines.
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
