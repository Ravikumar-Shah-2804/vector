"""Tests for vector.paper module: tables, significance, plots, orchestrator."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from vector.paper import (
    generate_all_artifacts,
    generate_latex_table3,
    generate_latex_table4,
    plot_f1_breakdown,
    plot_racs_scatter,
    save_figure,
    significance_tests,
)
from vector.results import ALL_DATASETS, ALL_METHODS


def _mock_results(n_datasets: int = 8) -> dict[str, dict[str, dict]]:
    """Build a synthetic results dict with n_datasets datasets and 4 methods."""
    datasets = ALL_DATASETS[:n_datasets]
    results: dict[str, dict[str, dict]] = {}
    for i, ds in enumerate(datasets):
        ds_results: dict[str, dict] = {}
        for j, method in enumerate(ALL_METHODS):
            f1 = 0.5 + 0.05 * j + 0.01 * i
            ds_results[method] = {
                "f1": round(f1, 4),
                "precision": round(f1 + 0.02, 4),
                "recall": round(f1 - 0.01, 4),
                "racs": round(f1 * 1.2, 4),
                "training_time": round(10.0 + j * 5.0, 1),
                "effective_size": 50 + j * 25,
            }
        results[ds] = ds_results
    return results


class TestGenerateLatexTable3:
    def test_structure(self, tmp_path):
        results = _mock_results()
        dummy = {"SWaT"}
        latex = generate_latex_table3(results, dummy, str(tmp_path))

        assert r"\begin{tabular}" in latex
        assert r"\hline" in latex
        assert r"\textbf{" in latex
        assert (tmp_path / "table3.tex").exists()

    def test_handles_na(self, tmp_path):
        results = _mock_results()
        # Set one entry to N/A
        results["NAB"]["Default"] = {
            "f1": "N/A", "precision": "N/A", "recall": "N/A",
            "racs": "N/A", "training_time": "N/A", "effective_size": "N/A",
        }
        latex = generate_latex_table3(results, set(), str(tmp_path))

        assert "--" in latex
        assert (tmp_path / "table3.tex").exists()


class TestGenerateLatexTable4:
    def test_structure(self, tmp_path):
        results = _mock_results()
        dummy = set()
        latex = generate_latex_table4(results, dummy, str(tmp_path))

        assert r"\begin{tabular}" in latex
        assert r"\hline" in latex
        assert r"\textbf{" in latex
        assert (tmp_path / "table4.tex").exists()


class TestSignificanceTests:
    def test_sufficient_pairs(self):
        results = _mock_results(8)
        test_results = significance_tests(results)

        # 3 baselines compared against VECTOR
        assert len(test_results) == 3
        for entry in test_results:
            assert isinstance(entry["p_value"], float)
            assert entry["n_pairs"] == 8

    def test_insufficient_pairs(self):
        results = _mock_results(3)
        test_results = significance_tests(results)

        for entry in test_results:
            assert entry["p_value"] is None
            assert "Insufficient" in entry.get("note", "")

    def test_zero_differences(self):
        """When VECTOR and a baseline have identical F1, p_value should be 1.0."""
        results = _mock_results(8)
        # Make Default identical to VECTOR
        for ds in ALL_DATASETS:
            if ds in results:
                results[ds]["Default"]["f1"] = results[ds]["VECTOR"]["f1"]

        test_results = significance_tests(results)
        default_entry = next(e for e in test_results if e["baseline"] == "Default")
        assert default_entry["p_value"] == 1.0


class TestSaveFigure:
    def test_dual_format(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_figure(fig, str(tmp_path), "test_fig")

        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.png").stat().st_size > 0
        assert (tmp_path / "test_fig.pdf").stat().st_size > 0


class TestPlotF1Breakdown:
    def test_creates_files(self, tmp_path):
        results = _mock_results()
        dummy = {"SWaT"}

        plot_f1_breakdown(results, dummy, str(tmp_path))

        assert (tmp_path / "f1_breakdown.png").exists()
        assert (tmp_path / "f1_breakdown.pdf").exists()


class TestPlotRacsScatter:
    def test_creates_files(self, tmp_path):
        results = _mock_results()

        plot_racs_scatter(results, str(tmp_path))

        assert (tmp_path / "racs_scatter.png").exists()
        assert (tmp_path / "racs_scatter.pdf").exists()


class TestGenerateAllArtifacts:
    def test_with_monkeypatch(self, tmp_path, monkeypatch):
        """Monkeypatch Optuna-dependent functions to no-ops and verify orchestrator."""
        import vector.paper as paper_mod

        monkeypatch.setattr(paper_mod, "plot_convergence", lambda *a, **kw: None)
        monkeypatch.setattr(paper_mod, "plot_pareto_evolution", lambda *a, **kw: None)
        monkeypatch.setattr(paper_mod, "plot_ablation", lambda *a, **kw: None)

        # Mock collect_results and is_dummy_data to avoid needing real data
        monkeypatch.setattr(
            paper_mod, "collect_results",
            lambda datasets, **kw: _mock_results(len(datasets)),
        )
        monkeypatch.setattr(paper_mod, "is_dummy_data", lambda ds, cfg: False)

        datasets = ALL_DATASETS
        dataset_config = {"datasets": {ds: {"raw_path": ""} for ds in datasets}}
        search_config = {}

        generate_all_artifacts(
            datasets, dataset_config, search_config,
            output_dir=str(tmp_path),
        )

        # Tables should be created
        assert (tmp_path / "table3.tex").exists()
        assert (tmp_path / "table4.tex").exists()
        assert (tmp_path / "significance.tex").exists()
        # Plots should be created
        assert (tmp_path / "f1_breakdown.png").exists()
        assert (tmp_path / "racs_scatter.png").exists()
