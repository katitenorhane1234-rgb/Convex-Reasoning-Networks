"""
tests/test_ablation.py
======================
Interface-level tests for :mod:`ablation`.

Verifies the *contracts* (return types, grid sizes, table format) of the
ablation study framework.  No training is executed — the tests verify the
structural scaffold only.
"""

from __future__ import annotations

from typing import Dict

import pytest

from config import CRNConfig, DataConfig, TrainConfig
from ablation import (
    AblationCell,
    AblationCellResult,
    AblationGroupResult,
    format_metric_ablation_table,
    format_solver_ablation_table,
    metric_ablation_grid,
    solver_ablation_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg() -> CRNConfig:
    cfg = CRNConfig(experiment_name="test_ablation")
    cfg.ablation.n_seeds = 2
    cfg.ablation.run_metric_ablation = True
    cfg.ablation.run_solver_ablation = True
    cfg.data.state_dim = 4
    cfg.data.n_trajectories = 20
    cfg.train.epochs = 1
    cfg.train.batch_size = 8
    return cfg


# ---------------------------------------------------------------------------
# AblationCell
# ---------------------------------------------------------------------------

class TestAblationCell:
    def test_construction(self) -> None:
        cell = AblationCell(
            name="SPD + Analytic",
            metric_type="spd",
            solver_name="analytic",
            seed=0,
        )
        assert cell.name == "SPD + Analytic"
        assert cell.metric_type == "spd"
        assert cell.solver_name == "analytic"
        assert cell.seed == 0


# ---------------------------------------------------------------------------
# metric_ablation_grid
# ---------------------------------------------------------------------------

class TestMetricAblationGrid:
    def test_grid_contains_both_metric_types(self, small_cfg: CRNConfig) -> None:
        cells = metric_ablation_grid(small_cfg)
        metric_types = {cell.metric_type for cell in cells}
        assert "spd" in metric_types
        assert "euclidean" in metric_types

    def test_grid_size_is_metrics_times_seeds(self, small_cfg: CRNConfig) -> None:
        cells = metric_ablation_grid(small_cfg)
        # 2 metric types × n_seeds
        assert len(cells) == 2 * small_cfg.ablation.n_seeds

    def test_solver_is_fixed_to_analytic(self, small_cfg: CRNConfig) -> None:
        cells = metric_ablation_grid(small_cfg)
        assert all(c.solver_name == "analytic" for c in cells)

    def test_each_cell_has_unique_seed_per_metric(self, small_cfg: CRNConfig) -> None:
        cells = metric_ablation_grid(small_cfg)
        spd_seeds = [c.seed for c in cells if c.metric_type == "spd"]
        assert len(spd_seeds) == len(set(spd_seeds))


# ---------------------------------------------------------------------------
# solver_ablation_grid
# ---------------------------------------------------------------------------

class TestSolverAblationGrid:
    def test_grid_contains_all_three_solvers(self, small_cfg: CRNConfig) -> None:
        cells = solver_ablation_grid(small_cfg)
        solver_names = {cell.solver_name for cell in cells}
        assert "analytic" in solver_names
        assert "pgd" in solver_names
        assert "frank_wolfe" in solver_names

    def test_grid_size_is_solvers_times_seeds(self, small_cfg: CRNConfig) -> None:
        cells = solver_ablation_grid(small_cfg)
        # 3 solver types × n_seeds
        assert len(cells) == 3 * small_cfg.ablation.n_seeds

    def test_each_cell_has_unique_seed_per_solver(self, small_cfg: CRNConfig) -> None:
        cells = solver_ablation_grid(small_cfg)
        analytic_seeds = [c.seed for c in cells if c.solver_name == "analytic"]
        assert len(analytic_seeds) == len(set(analytic_seeds))


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _make_group_result(name: str, metric_type: str, solver_name: str) -> AblationGroupResult:
    return AblationGroupResult(
        name=name,
        metric_type=metric_type,
        solver_name=solver_name,
        n_seeds=3,
        mean_test_loss=0.05,
        std_test_loss=0.002,
        mean_val_loss=0.04,
        std_val_loss=0.001,
        mean_condition_number=5.0,
        std_condition_number=0.5,
        mean_success_rate=0.95,
        std_success_rate=0.02,
        mean_forward_ms=1.2,
        std_forward_ms=0.1,
        mean_solver_iter=20.0,
        std_solver_iter=3.0,
    )


class TestFormatMetricAblationTable:
    def test_returns_string(self) -> None:
        results = {
            "SPD": _make_group_result("SPD", "spd", "analytic"),
            "Euclidean": _make_group_result("Euclidean", "euclidean", "analytic"),
        }
        table = format_metric_ablation_table(results)
        assert isinstance(table, str)

    def test_contains_condition_column(self) -> None:
        results = {"SPD": _make_group_result("SPD", "spd", "analytic")}
        table = format_metric_ablation_table(results)
        # Should mention condition number or κ
        assert any(kw in table.lower() for kw in ("condition", "kappa", "κ"))

    def test_contains_all_group_names(self) -> None:
        results = {
            "SPD": _make_group_result("SPD", "spd", "analytic"),
            "Euclidean": _make_group_result("Euclidean", "euclidean", "analytic"),
        }
        table = format_metric_ablation_table(results)
        assert "SPD" in table
        assert "Euclidean" in table


class TestFormatSolverAblationTable:
    def test_returns_string(self) -> None:
        results = {
            "Analytic": _make_group_result("Analytic", "spd", "analytic"),
            "PGD": _make_group_result("PGD", "spd", "pgd"),
            "FW": _make_group_result("FW", "spd", "frank_wolfe"),
        }
        table = format_solver_ablation_table(results)
        assert isinstance(table, str)

    def test_contains_solver_names(self) -> None:
        results = {
            "Analytic": _make_group_result("Analytic", "spd", "analytic"),
            "PGD": _make_group_result("PGD", "spd", "pgd"),
        }
        table = format_solver_ablation_table(results)
        assert "Analytic" in table
        assert "PGD" in table

    def test_contains_iteration_column(self) -> None:
        results = {"Analytic": _make_group_result("Analytic", "spd", "analytic")}
        table = format_solver_ablation_table(results)
        assert any(kw in table.lower() for kw in ("iter", "iterations"))


# ---------------------------------------------------------------------------
# AblationGroupResult helpers
# ---------------------------------------------------------------------------

class TestAblationGroupResult:
    def test_to_dict_has_required_keys(self) -> None:
        gr = _make_group_result("SPD", "spd", "analytic")
        d = gr.to_dict()
        for key in ("name", "metric_type", "solver_name", "n_seeds",
                    "mean_test_loss", "std_test_loss", "mean_success_rate"):
            assert key in d

    def test_summary_row_returns_string(self) -> None:
        gr = _make_group_result("SPD", "spd", "analytic")
        row = gr.summary_row()
        assert isinstance(row, str)
        assert len(row) > 0
