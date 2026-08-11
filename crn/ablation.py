"""
ablation.py
===========
Ablation study framework for Convex Reasoning Networks.

Runs a structured grid of experiments that isolate the contribution of each
architectural component:

**Metric ablation** — compares the learnable SPD metric against the fixed
Euclidean (identity) metric, holding everything else constant.

**Solver ablation** — compares the three solver implementations (Analytic,
PGD, Frank-Wolfe) on identical model instances, measuring both solution
quality and computational cost.

Each ablation cell is repeated over multiple seeds to obtain mean ± std
estimates.  Results are automatically saved as:

* ``results/<experiment_name>/ablation_metric.json``
* ``results/<experiment_name>/ablation_solver.json``
* ``results/<experiment_name>/ablation_table_metric.txt``
* ``results/<experiment_name>/ablation_table_solver.txt``

The primary entry point is :func:`run_ablation_study`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from config import CRNConfig, RESULTS_DIR
from crn import build_crn
from dataset import build_dataloaders
from evaluate import EvaluationReport, evaluate, compute_metric_stats
from train import train
from utils import set_seed


# ---------------------------------------------------------------------------
# Ablation cell and result types
# ---------------------------------------------------------------------------


@dataclass
class AblationCell:
    """
    A single experimental condition in the ablation grid.

    Attributes
    ----------
    name:
        Human-readable label for this cell (e.g. ``'SPD + Analytic'``).
    metric_type:
        One of ``'spd'`` or ``'euclidean'``.
    solver_name:
        One of ``'analytic'``, ``'pgd'``, ``'frank_wolfe'``.
    seed:
        Random seed for this run.
    """

    name: str
    metric_type: str
    solver_name: str
    seed: int


@dataclass
class AblationCellResult:
    """Results for a single ablation cell (one seed)."""

    cell: AblationCell
    test_loss: float
    val_loss: float
    condition_number: float
    spectral_norm_A: float
    mean_convergence_rate: float
    success_rate: float
    mean_forward_ms: float
    mean_solver_iter: float
    total_train_time_s: float
    best_epoch: int
    stopped_early: bool


@dataclass
class AblationGroupResult:
    """
    Aggregated results for one ablation condition across multiple seeds.

    Statistics (mean ± std) are computed over the seed dimension.
    """

    name: str
    metric_type: str
    solver_name: str
    n_seeds: int

    mean_test_loss: float
    std_test_loss: float

    mean_val_loss: float
    std_val_loss: float

    mean_condition_number: float
    std_condition_number: float

    mean_success_rate: float
    std_success_rate: float

    mean_forward_ms: float
    std_forward_ms: float

    mean_solver_iter: float
    std_solver_iter: float

    seed_results: List[AblationCellResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary."""
        d = {
            "name": self.name,
            "metric_type": self.metric_type,
            "solver_name": self.solver_name,
            "n_seeds": self.n_seeds,
            "mean_test_loss": self.mean_test_loss,
            "std_test_loss": self.std_test_loss,
            "mean_val_loss": self.mean_val_loss,
            "std_val_loss": self.std_val_loss,
            "mean_condition_number": self.mean_condition_number,
            "std_condition_number": self.std_condition_number,
            "mean_success_rate": self.mean_success_rate,
            "std_success_rate": self.std_success_rate,
            "mean_forward_ms": self.mean_forward_ms,
            "std_forward_ms": self.std_forward_ms,
            "mean_solver_iter": self.mean_solver_iter,
            "std_solver_iter": self.std_solver_iter,
        }
        return d

    def summary_row(self, width: int = 14) -> str:
        """
        Return a single formatted ASCII table row for this condition.

        Parameters
        ----------
        width:
            Column width for numeric fields.
        """
        return (
            f"{self.name:<20} "
            f"{self.mean_test_loss:{width}.6f}±{self.std_test_loss:.4f}  "
            f"{self.mean_condition_number:{width}.2f}±{self.std_condition_number:.2f}  "
            f"{self.mean_success_rate:{width}.4f}±{self.std_success_rate:.4f}  "
            f"{self.mean_forward_ms:{width}.3f}ms"
        )


# ---------------------------------------------------------------------------
# Ablation grid definitions
# ---------------------------------------------------------------------------


def metric_ablation_grid(cfg: CRNConfig) -> List[AblationCell]:
    """
    Build the ablation grid for the metric comparison experiment.

    Returns one cell per (metric_type × seed) combination, holding the
    solver fixed at ``'analytic'``.

    Parameters
    ----------
    cfg:
        Experiment configuration (provides ``n_seeds`` and ``solver``).

    Returns
    -------
    list[AblationCell]
    """
    metric_types = ["spd", "euclidean"]
    cells: List[AblationCell] = []
    for metric_type in metric_types:
        label = "SPD" if metric_type == "spd" else "Euclidean"
        for s in range(cfg.ablation.n_seeds):
            cells.append(AblationCell(
                name=f"{label} (seed={s})",
                metric_type=metric_type,
                solver_name="analytic",
                seed=s,
            ))
    return cells


def solver_ablation_grid(cfg: CRNConfig) -> List[AblationCell]:
    """
    Build the ablation grid for the solver comparison experiment.

    Returns one cell per (solver_name × seed) combination, holding the
    metric fixed at the value specified in ``cfg.model.metric_type``.

    Parameters
    ----------
    cfg:
        Experiment configuration.

    Returns
    -------
    list[AblationCell]
    """
    solver_names = ["analytic", "pgd", "frank_wolfe"]
    solver_labels = {"analytic": "Analytic", "pgd": "PGD", "frank_wolfe": "FrankWolfe"}
    cells: List[AblationCell] = []
    for solver_name in solver_names:
        for s in range(cfg.ablation.n_seeds):
            cells.append(AblationCell(
                name=f"{solver_labels[solver_name]} (seed={s})",
                metric_type=cfg.model.metric_type,
                solver_name=solver_name,
                seed=s,
            ))
    return cells


# ---------------------------------------------------------------------------
# Single-cell execution
# ---------------------------------------------------------------------------


def run_cell(
    base_cfg: CRNConfig,
    cell: AblationCell,
    verbose: bool = False,
) -> AblationCellResult:
    """
    Train and evaluate a single ablation cell.

    Constructs a modified copy of ``base_cfg`` with the cell's metric and
    solver, trains from scratch, and evaluates on the test set.

    Parameters
    ----------
    base_cfg:
        Base configuration to modify (not modified in-place).
    cell:
        Ablation cell specifying the condition and seed.
    verbose:
        If True, print progress to stdout.

    Returns
    -------
    AblationCellResult
    """
    cfg = _make_cell_config(base_cfg, cell)

    if verbose:
        print(f"  Running cell: {cell.name} | metric={cell.metric_type} | solver={cell.solver_name} | seed={cell.seed}")

    t0 = time.monotonic()
    model, train_result = train(cfg)
    train_time = time.monotonic() - t0

    report = evaluate(cfg, model=model)

    return AblationCellResult(
        cell=cell,
        test_loss=report.test_loss,
        val_loss=report.val_loss,
        condition_number=report.metric_stats.condition_number,
        spectral_norm_A=report.metric_stats.spectral_norm_A,
        mean_convergence_rate=report.solver_stats.convergence_rate,
        success_rate=report.convergence_stats.success_rate,
        mean_forward_ms=report.timing_stats.mean_forward_ms,
        mean_solver_iter=report.solver_stats.mean_n_iter,
        total_train_time_s=train_time,
        best_epoch=train_result.best_epoch,
        stopped_early=train_result.stopped_early,
    )


def _make_cell_config(base_cfg: CRNConfig, cell: AblationCell) -> CRNConfig:
    """
    Derive a configuration for a single ablation cell.

    Creates a deep copy of ``base_cfg`` and overrides metric type, solver
    name, random seed, and experiment name.

    Parameters
    ----------
    base_cfg:
        Template configuration.
    cell:
        Ablation cell specification.

    Returns
    -------
    CRNConfig
    """
    cfg = copy.deepcopy(base_cfg)
    cfg.model.metric_type = cell.metric_type
    cfg.model.solver = cell.solver_name
    cfg.train.seed = cell.seed
    cfg.data.seed = cell.seed
    # Give each cell a unique experiment name to avoid checkpoint conflicts
    safe_name = cell.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    cfg.experiment_name = (
        f"{base_cfg.experiment_name}_ablation_{safe_name}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Grid aggregation
# ---------------------------------------------------------------------------


def aggregate_group(
    name: str,
    metric_type: str,
    solver_name: str,
    cell_results: List[AblationCellResult],
) -> AblationGroupResult:
    """
    Aggregate a list of single-seed results into mean ± std statistics.

    Parameters
    ----------
    name:
        Group label.
    metric_type:
        Metric type shared by all cells.
    solver_name:
        Solver name shared by all cells.
    cell_results:
        List of per-seed results.

    Returns
    -------
    AblationGroupResult
    """
    import statistics as _stats

    def _mean(vals): return _stats.mean(vals) if vals else 0.0
    def _std(vals): return _stats.stdev(vals) if len(vals) > 1 else 0.0

    test_losses = [r.test_loss for r in cell_results]
    val_losses = [r.val_loss for r in cell_results]
    cond_nums = [r.condition_number for r in cell_results]
    success_rates = [r.success_rate for r in cell_results]
    fwd_ms = [r.mean_forward_ms for r in cell_results]
    solver_iters = [r.mean_solver_iter for r in cell_results]

    return AblationGroupResult(
        name=name,
        metric_type=metric_type,
        solver_name=solver_name,
        n_seeds=len(cell_results),
        mean_test_loss=_mean(test_losses),
        std_test_loss=_std(test_losses),
        mean_val_loss=_mean(val_losses),
        std_val_loss=_std(val_losses),
        mean_condition_number=_mean(cond_nums),
        std_condition_number=_std(cond_nums),
        mean_success_rate=_mean(success_rates),
        std_success_rate=_std(success_rates),
        mean_forward_ms=_mean(fwd_ms),
        std_forward_ms=_std(fwd_ms),
        mean_solver_iter=_mean(solver_iters),
        std_solver_iter=_std(solver_iters),
        seed_results=cell_results,
    )


def run_metric_ablation(
    cfg: CRNConfig,
    verbose: bool = True,
) -> Dict[str, AblationGroupResult]:
    """
    Run the full metric ablation study.

    Executes all (metric_type × seed) cells and aggregates results.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    verbose:
        Print per-cell progress.

    Returns
    -------
    dict
        Mapping from group name to :class:`AblationGroupResult`.
    """
    grid = metric_ablation_grid(cfg)

    # Group by metric_type
    groups: Dict[str, List[AblationCell]] = {}
    for cell in grid:
        key = cell.metric_type
        groups.setdefault(key, []).append(cell)

    results: Dict[str, AblationGroupResult] = {}
    for metric_type, cells in groups.items():
        cell_results = []
        for cell in cells:
            cr = run_cell(cfg, cell, verbose=verbose)
            cell_results.append(cr)
        label = "SPD" if metric_type == "spd" else "Euclidean"
        results[label] = aggregate_group(label, metric_type, "analytic", cell_results)

    return results


def run_solver_ablation(
    cfg: CRNConfig,
    verbose: bool = True,
) -> Dict[str, AblationGroupResult]:
    """
    Run the full solver ablation study.

    Executes all (solver_name × seed) cells and aggregates results.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    verbose:
        Print per-cell progress.

    Returns
    -------
    dict
        Mapping from group name to :class:`AblationGroupResult`.
    """
    grid = solver_ablation_grid(cfg)

    # Group by solver_name
    groups: Dict[str, List[AblationCell]] = {}
    for cell in grid:
        key = cell.solver_name
        groups.setdefault(key, []).append(cell)

    label_map = {"analytic": "Analytic", "pgd": "PGD", "frank_wolfe": "FrankWolfe"}
    results: Dict[str, AblationGroupResult] = {}
    for solver_name, cells in groups.items():
        cell_results = []
        for cell in cells:
            cr = run_cell(cfg, cell, verbose=verbose)
            cell_results.append(cr)
        label = label_map.get(solver_name, solver_name)
        results[label] = aggregate_group(label, cfg.model.metric_type, solver_name, cell_results)

    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def format_metric_ablation_table(
    results: Dict[str, AblationGroupResult],
) -> str:
    """
    Format the metric ablation results as a LaTeX-ready ASCII table.

    Columns: Condition | Test Loss (↓) | Val Loss (↓) | κ(M) | Success Rate (↑) | Time (ms)

    Parameters
    ----------
    results:
        Output of :func:`run_metric_ablation`.

    Returns
    -------
    str
        Multi-line ASCII table string.
    """
    c = 14
    header = (
        f"{'Condition':<20} "
        f"{'Test Loss (↓)':>{c}} "
        f"{'Val Loss (↓)':>{c}} "
        f"{'κ(M)':>{c}} "
        f"{'Succ. Rate (↑)':>{c}} "
        f"{'Time (ms)':>{c}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, gr in results.items():
        lines.append(
            f"{name:<20} "
            f"{gr.mean_test_loss:{c}.6f} "
            f"{gr.mean_val_loss:{c}.6f} "
            f"{gr.mean_condition_number:{c}.2f} "
            f"{gr.mean_success_rate:{c}.4f} "
            f"{gr.mean_forward_ms:{c}.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_solver_ablation_table(
    results: Dict[str, AblationGroupResult],
) -> str:
    """
    Format the solver ablation results as a LaTeX-ready ASCII table.

    Columns: Solver | Test Loss (↓) | Iterations | Convergence Rate | Time (ms/step)

    Parameters
    ----------
    results:
        Output of :func:`run_solver_ablation`.

    Returns
    -------
    str
        Multi-line ASCII table string.
    """
    c = 16
    header = (
        f"{'Solver':<14} "
        f"{'Test Loss (↓)':>{c}} "
        f"{'Iterations':>{c}} "
        f"{'Conv. Rate':>{c}} "
        f"{'Time (ms/step)':>{c}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, gr in results.items():
        lines.append(
            f"{name:<14} "
            f"{gr.mean_test_loss:{c}.6f} "
            f"{gr.mean_solver_iter:{c}.2f} "
            f"{gr.mean_success_rate:{c}.4f} "
            f"{gr.mean_forward_ms:{c}.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_ablation_results(
    results: Dict[str, AblationGroupResult],
    path: Path,
) -> None:
    """
    Save ablation results to a JSON file.

    Parameters
    ----------
    results:
        Ablation results dict.
    path:
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {name: gr.to_dict() for name, gr in results.items()}
    path.write_text(json.dumps(serialisable, indent=2, default=str))


def load_ablation_results(path: Path) -> Dict[str, dict]:
    """
    Load ablation results from a previously saved JSON file.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    dict
        Raw dictionary (de-serialise with :class:`AblationGroupResult` if needed).
    """
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def run_ablation_study(
    cfg: CRNConfig,
    verbose: bool = True,
) -> Tuple[Dict[str, AblationGroupResult], Dict[str, AblationGroupResult]]:
    """
    Run the complete ablation study (metric + solver).

    Saves results and formatted tables to the ``results/`` directory.

    Parameters
    ----------
    cfg:
        Experiment configuration.  Controls which ablations are enabled via
        ``cfg.ablation.run_metric_ablation`` and
        ``cfg.ablation.run_solver_ablation``.
    verbose:
        Print progress and tables to stdout.

    Returns
    -------
    tuple
        ``(metric_results, solver_results)`` — each is a dict from condition
        name to :class:`AblationGroupResult`.  Empty dict if the corresponding
        ablation is disabled.
    """
    out_dir = RESULTS_DIR / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_results: Dict[str, AblationGroupResult] = {}
    solver_results: Dict[str, AblationGroupResult] = {}

    if cfg.ablation.run_metric_ablation:
        if verbose:
            print("\n=== Metric Ablation ===")
        metric_results = run_metric_ablation(cfg, verbose=verbose)
        save_ablation_results(metric_results, out_dir / "ablation_metric.json")
        table_m = format_metric_ablation_table(metric_results)
        (out_dir / "ablation_table_metric.txt").write_text(table_m)
        if verbose:
            print(table_m)

    if cfg.ablation.run_solver_ablation:
        if verbose:
            print("\n=== Solver Ablation ===")
        solver_results = run_solver_ablation(cfg, verbose=verbose)
        save_ablation_results(solver_results, out_dir / "ablation_solver.json")
        table_s = format_solver_ablation_table(solver_results)
        (out_dir / "ablation_table_solver.txt").write_text(table_s)
        if verbose:
            print(table_s)

    return metric_results, solver_results
