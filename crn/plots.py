
"""
plots.py
========
Publication-quality figure generation for Convex Reasoning Networks.

All figures are saved in both PDF (vector, for LaTeX inclusion) and PNG
(raster, for quick inspection) formats under ``figures/<experiment_name>/``.

Figure catalogue
----------------

+---------------------------+-------------------------------------------+
| Figure                    | Function                                  |
+===========================+===========================================+
| Training loss curve       | :func:`plot_training_loss`                |
| Validation loss curve     | :func:`plot_validation_loss`              |
| Train + val combined      | :func:`plot_loss_curves`                  |
| Convergence curves        | :func:`plot_convergence_curves`           |
| Runtime comparison        | :func:`plot_runtime_comparison`           |
| Ablation — metric         | :func:`plot_metric_ablation`              |
| Ablation — solver         | :func:`plot_solver_ablation`              |
| Metric eigenvalue spectra | :func:`plot_eigenvalue_spectrum`          |
| Condition number κ(M)     | :func:`plot_condition_number`             |
| Trajectory comparison     | :func:`plot_trajectory_comparison`        |
| All figures (batch)       | :func:`generate_all_figures`             |
+---------------------------+-------------------------------------------+

Style conventions
-----------------
* Uses the ``seaborn-v0_8-paper`` matplotlib style for publication readability.
* Font: serif (Latin Modern Roman family, matching LaTeX defaults).
* Figure sizes default to single-column (3.5 in) or double-column (7.16 in).
* Colour palette: colourblind-safe ``tableau-colorblind10``.
* All axes use ≥ 10 pt font for readability in print.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from config import CRNConfig, FIGURES_DIR
from evaluate import EvaluationReport
from train import EpochMetrics, TrainingResult


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

# Use a non-interactive backend when running on headless servers
matplotlib.use("Agg")

SINGLE_COL_WIDTH: float = 3.5   # inches (one IEEE/NeurIPS column)
DOUBLE_COL_WIDTH: float = 7.16  # inches (two columns / full width)
DEFAULT_DPI: int = 300           # raster export resolution

COLOUR_PALETTE: List[str] = [
    "#006BA4", "#FF800E", "#ABABAB", "#595959",
    "#5F9ED1", "#C85200", "#898989", "#A2C8EC",
]

# Marker styles for distinguishable lines (colourblind + grayscale safe)
MARKERS: List[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _apply_global_style() -> None:
    """Apply consistent publication-quality matplotlib style."""
    # Try the seaborn paper style; fall back gracefully
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass  # Use matplotlib defaults if neither is available

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_figure(fig: Figure, path: Path, dpi: int = DEFAULT_DPI) -> None:
    """
    Save a figure as both PDF and PNG.

    Parameters
    ----------
    fig:
        The matplotlib figure to save.
    path:
        Base path without extension (e.g. ``figures/exp1/training_loss``).
        The function appends ``.pdf`` and ``.png`` automatically.
    dpi:
        Resolution for the PNG export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path) + ".png", dpi=dpi, bbox_inches="tight")


def _make_figure(
    ncols: int = 1,
    nrows: int = 1,
    width: float = SINGLE_COL_WIDTH,
    height: Optional[float] = None,
) -> Tuple[Figure, Axes | np.ndarray]:
    """
    Create a figure with sensible defaults.

    Parameters
    ----------
    ncols, nrows:
        Grid dimensions.
    width:
        Figure width in inches.
    height:
        Figure height in inches.  Defaults to ``width * nrows / ncols * 0.75``.

    Returns
    -------
    tuple
        ``(fig, ax)`` — ax is a single Axes if ncols == nrows == 1, else an ndarray.
    """
    _apply_global_style()
    if height is None:
        height = width * (nrows / ncols) * 0.75

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    return fig, axes


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def plot_training_loss(
    result: TrainingResult,
    cfg: CRNConfig,
    *,
    log_scale: bool = True,
    save: bool = True,
) -> Figure:
    """
    Plot the per-epoch training loss curve.

    Parameters
    ----------
    result:
        :class:`~train.TrainingResult` from a completed training run.
    cfg:
        Experiment configuration (used for titles and save paths).
    log_scale:
        If True, use a log-scale y-axis.
    save:
        If True, save to ``figures/<experiment_name>/training_loss.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure()
    epochs = [em.epoch for em in result.epoch_metrics]
    losses = [em.train_loss for em in result.epoch_metrics]

    ax.plot(epochs, losses, color=COLOUR_PALETTE[0], linewidth=1.5,
            marker="", label="Train loss")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — {cfg.experiment_name}")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "training_loss")

    return fig


def plot_validation_loss(
    result: TrainingResult,
    cfg: CRNConfig,
    *,
    log_scale: bool = True,
    save: bool = True,
) -> Figure:
    """
    Plot the per-epoch validation loss curve.

    Parameters
    ----------
    result:
        Training result.
    cfg:
        Experiment configuration.
    log_scale:
        If True, use a log-scale y-axis.
    save:
        If True, save figure.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure()
    epochs = [em.epoch for em in result.epoch_metrics]
    losses = [em.val_loss for em in result.epoch_metrics]

    ax.plot(epochs, losses, color=COLOUR_PALETTE[1], linewidth=1.5, label="Val loss")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Validation Loss — {cfg.experiment_name}")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "validation_loss")

    return fig


def plot_loss_curves(
    result: TrainingResult,
    cfg: CRNConfig,
    *,
    log_scale: bool = True,
    save: bool = True,
) -> Figure:
    """
    Plot training and validation loss on a single set of axes.

    Marks the best-validation-loss epoch with a vertical dashed line and a
    star annotation.

    Parameters
    ----------
    result:
        Training result.
    cfg:
        Experiment configuration.
    log_scale:
        If True, use a log-scale y-axis.
    save:
        If True, save to ``figures/<experiment_name>/loss_curves.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure(width=DOUBLE_COL_WIDTH)
    epochs = [em.epoch for em in result.epoch_metrics]
    train_losses = [em.train_loss for em in result.epoch_metrics]
    val_losses = [em.val_loss for em in result.epoch_metrics]

    ax.plot(epochs, train_losses, color=COLOUR_PALETTE[0], linewidth=1.5,
            label="Train", marker="", alpha=0.85)
    ax.plot(epochs, val_losses, color=COLOUR_PALETTE[1], linewidth=1.5,
            label="Validation", marker="", linestyle="--", alpha=0.85)

    # Mark best epoch
    if result.epoch_metrics:
        best_ep = result.best_epoch
        best_val = result.best_val_loss
        ax.axvline(x=best_ep, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.scatter([best_ep], [best_val], marker="*", s=80,
                   color=COLOUR_PALETTE[1], zorder=5, label=f"Best (ep {best_ep})")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curves — {cfg.experiment_name}")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "loss_curves")

    return fig


# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------


def plot_convergence_curves(
    lyapunov_exponents: Sequence[Sequence[float]],
    cfg: CRNConfig,
    *,
    n_display: int = 20,
    save: bool = True,
) -> Figure:
    """
    Plot per-trajectory Lyapunov exponent convergence curves.

    Shows a random subset of ``n_display`` trajectories as faint grey lines
    with the mean trajectory highlighted in colour.

    Parameters
    ----------
    lyapunov_exponents:
        List of per-trajectory exponent sequences (variable length).
    cfg:
        Experiment configuration.
    n_display:
        Number of individual trajectories to overlay.
    save:
        If True, save to ``figures/<experiment_name>/convergence_curves.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure(width=DOUBLE_COL_WIDTH)

    if not lyapunov_exponents:
        ax.set_title("No convergence data")
        return fig

    # Pad to common length
    max_len = max(len(seq) for seq in lyapunov_exponents)
    padded = [list(seq) + [float("nan")] * (max_len - len(seq))
              for seq in lyapunov_exponents]
    arr = np.array(padded, dtype=float)          # (N, T)

    # Subsample for display
    rng = np.random.default_rng(0)
    n_traj = arr.shape[0]
    display_idx = rng.choice(n_traj, size=min(n_display, n_traj), replace=False)

    steps = np.arange(1, max_len + 1)
    for idx in display_idx:
        ax.plot(steps, arr[idx], color="grey", linewidth=0.6, alpha=0.3)

    # Mean trajectory (ignoring NaNs)
    mean_curve = np.nanmean(arr, axis=0)
    ax.plot(steps, mean_curve, color=COLOUR_PALETTE[0], linewidth=2.0,
            label="Mean Lyapunov exp.", zorder=3)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Neutral (λ=0)")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Lyapunov exponent λ")
    ax.set_title(f"Convergence Curves — {cfg.experiment_name}")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "convergence_curves")

    return fig


# ---------------------------------------------------------------------------
# Runtime comparison
# ---------------------------------------------------------------------------


def plot_runtime_comparison(
    solver_times: Dict[str, List[float]],
    cfg: CRNConfig,
    *,
    save: bool = True,
) -> Figure:
    """
    Bar chart comparing per-solver mean inference times.

    Displays mean ± std as error bars.  The x-axis shows solver names;
    the y-axis shows milliseconds per projection.

    Parameters
    ----------
    solver_times:
        Dict mapping solver name → list of per-call times (ms).
    cfg:
        Experiment configuration.
    save:
        If True, save to ``figures/<experiment_name>/runtime_comparison.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure()

    names = list(solver_times.keys())
    means = [float(np.mean(t)) for t in solver_times.values()]
    stds = [float(np.std(t)) for t in solver_times.values()]
    colours = COLOUR_PALETTE[:len(names)]

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2},
                  width=0.55, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Time (ms / projection)")
    ax.set_title(f"Solver Runtime — {cfg.experiment_name}")
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "runtime_comparison")

    return fig


# ---------------------------------------------------------------------------
# Ablation figures
# ---------------------------------------------------------------------------


def plot_metric_ablation(
    ablation_results: Dict[str, object],
    cfg: CRNConfig,
    *,
    save: bool = True,
) -> Figure:
    """
    Grouped bar chart for the metric ablation study.

    Shows mean ± std test loss and condition number for each metric type.

    Parameters
    ----------
    ablation_results:
        Output of :func:`~ablation.run_metric_ablation`.
    cfg:
        Experiment configuration.
    save:
        If True, save to ``figures/<experiment_name>/ablation_metric.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, axes = _make_figure(ncols=2, width=DOUBLE_COL_WIDTH)
    ax_loss, ax_kappa = axes  # type: ignore[misc]

    names = list(ablation_results.keys())
    grs = list(ablation_results.values())
    x = np.arange(len(names))
    colours = COLOUR_PALETTE[:len(names)]

    # Test loss
    means_loss = [gr.mean_test_loss for gr in grs]      # type: ignore[attr-defined]
    stds_loss = [gr.std_test_loss for gr in grs]        # type: ignore[attr-defined]
    ax_loss.bar(x, means_loss, yerr=stds_loss, capsize=4, color=colours,
                width=0.55, edgecolor="white")
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(names)
    ax_loss.set_ylabel("Test Loss (MSE)")
    ax_loss.set_title("Test Loss")

    # Condition number
    means_k = [gr.mean_condition_number for gr in grs]  # type: ignore[attr-defined]
    stds_k = [gr.std_condition_number for gr in grs]    # type: ignore[attr-defined]
    ax_kappa.bar(x, means_k, yerr=stds_k, capsize=4, color=colours,
                 width=0.55, edgecolor="white")
    ax_kappa.set_xticks(x)
    ax_kappa.set_xticklabels(names)
    ax_kappa.set_ylabel("Condition Number κ(M)")
    ax_kappa.set_title("Metric Condition Number")

    fig.suptitle(f"Metric Ablation — {cfg.experiment_name}", y=1.02)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "ablation_metric")

    return fig


def plot_solver_ablation(
    ablation_results: Dict[str, object],
    cfg: CRNConfig,
    *,
    save: bool = True,
) -> Figure:
    """
    Multi-panel figure for the solver ablation study.

    Panel 1: Mean test loss per solver (bar chart).
    Panel 2: Mean iterations per solve call (bar chart).
    Panel 3: Mean inference time per projection (bar chart).

    Parameters
    ----------
    ablation_results:
        Output of :func:`~ablation.run_solver_ablation`.
    cfg:
        Experiment configuration.
    save:
        If True, save to ``figures/<experiment_name>/ablation_solver.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, axes = _make_figure(ncols=3, width=DOUBLE_COL_WIDTH)
    ax_loss, ax_iter, ax_time = axes  # type: ignore[misc]

    names = list(ablation_results.keys())
    grs = list(ablation_results.values())
    x = np.arange(len(names))
    colours = COLOUR_PALETTE[:len(names)]

    def _bar(ax, vals, stds, ylabel, title):
        ax.bar(x, vals, yerr=stds, capsize=4, color=colours, width=0.55, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    _bar(ax_loss,
         [gr.mean_test_loss for gr in grs],      # type: ignore[attr-defined]
         [gr.std_test_loss for gr in grs],        # type: ignore[attr-defined]
         "Test Loss (MSE)", "Test Loss")
    _bar(ax_iter,
         [gr.mean_solver_iter for gr in grs],     # type: ignore[attr-defined]
         [gr.std_solver_iter for gr in grs],      # type: ignore[attr-defined]
         "Iterations / solve", "Solver Iterations")
    _bar(ax_time,
         [gr.mean_forward_ms for gr in grs],      # type: ignore[attr-defined]
         [gr.std_forward_ms for gr in grs],       # type: ignore[attr-defined]
         "Time (ms / fwd pass)", "Forward-Pass Time")

    fig.suptitle(f"Solver Ablation — {cfg.experiment_name}", y=1.02)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "ablation_solver")

    return fig


# ---------------------------------------------------------------------------
# Metric spectral figures
# ---------------------------------------------------------------------------


def plot_eigenvalue_spectrum(
    report: EvaluationReport,
    cfg: CRNConfig,
    *,
    save: bool = True,
) -> Figure:
    """
    Stem plot of the eigenvalue spectrum of the learned metric M.

    Shows each eigenvalue as a vertical stem, annotated with the condition
    number κ(M) and the regularisation floor ε.

    Parameters
    ----------
    report:
        Evaluation report containing ``metric_stats.eigenvalues``.
    cfg:
        Experiment configuration.
    save:
        If True, save to ``figures/<experiment_name>/eigenvalue_spectrum.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure(width=SINGLE_COL_WIDTH)

    eigs = sorted(report.metric_stats.eigenvalues)
    x = np.arange(1, len(eigs) + 1)

    markerline, stemlines, baseline = ax.stem(
        x, eigs, linefmt="-", markerfmt="o", basefmt="k-"
    )
    markerline.set_markersize(4)
    markerline.set_color(COLOUR_PALETTE[0])
    stemlines.set_linewidth(1.0)
    stemlines.set_color(COLOUR_PALETTE[0])

    # Annotate condition number
    kappa = report.metric_stats.condition_number
    ax.text(0.97, 0.97,
            f"κ(M) = {kappa:.2f}\nlog κ = {report.metric_stats.log_condition_number:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5))

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title(f"Metric Eigenvalue Spectrum — {cfg.experiment_name}")
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "eigenvalue_spectrum")

    return fig


def plot_condition_number(
    epoch_metrics: List[EpochMetrics],
    cfg: CRNConfig,
    *,
    log_scale: bool = True,
    save: bool = True,
) -> Figure:
    """
    Line chart of the condition number κ(M) over the course of training.

    Parameters
    ----------
    epoch_metrics:
        Per-epoch training metrics containing ``condition_number_M``.
    cfg:
        Experiment configuration.
    log_scale:
        If True, use a log-scale y-axis.
    save:
        If True, save to ``figures/<experiment_name>/condition_number.{pdf,png}``.

    Returns
    -------
    Figure
    """
    fig, ax = _make_figure(width=DOUBLE_COL_WIDTH)

    epochs = [em.epoch for em in epoch_metrics]
    kappas = [em.condition_number_M for em in epoch_metrics]

    ax.plot(epochs, kappas, color=COLOUR_PALETTE[2], linewidth=1.5,
            label="κ(M)")
    if log_scale and all(k > 0 for k in kappas):
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Condition number κ(M)")
    ax.set_title(f"Metric Condition Number — {cfg.experiment_name}")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "condition_number")

    return fig


# ---------------------------------------------------------------------------
# Trajectory comparison
# ---------------------------------------------------------------------------


def plot_trajectory_comparison(
    predicted: np.ndarray,
    target: np.ndarray,
    cfg: CRNConfig,
    *,
    n_dims: int = 3,
    trajectory_idx: int = 0,
    save: bool = True,
) -> Figure:
    """
    Line chart comparing predicted and ground-truth trajectories.

    Plots the first ``n_dims`` state dimensions as separate sub-panels.

    Parameters
    ----------
    predicted:
        Predicted state trajectory, shape ``(T+1, state_dim)`` (single trajectory).
    target:
        Ground-truth trajectory, same shape.
    cfg:
        Experiment configuration.
    n_dims:
        Number of state dimensions to plot.
    trajectory_idx:
        Index of the trajectory (used only in the plot title).
    save:
        If True, save to ``figures/<experiment_name>/trajectory_comparison.{pdf,png}``.

    Returns
    -------
    Figure
    """
    n_dims = min(n_dims, predicted.shape[-1])
    fig, axes = _make_figure(nrows=n_dims, ncols=1, width=DOUBLE_COL_WIDTH,
                             height=n_dims * 1.5)

    if n_dims == 1:
        axes = [axes]

    T = predicted.shape[0]
    steps = np.arange(T)

    for d_idx, ax in enumerate(axes):
        ax.plot(steps, target[:, d_idx], color=COLOUR_PALETTE[0],
                linewidth=1.5, label="Ground truth", linestyle="-")
        ax.plot(steps, predicted[:, d_idx], color=COLOUR_PALETTE[1],
                linewidth=1.5, label="Predicted", linestyle="--")
        ax.set_ylabel(f"$x_{{{d_idx}}}$")
        if d_idx == 0:
            ax.legend(frameon=False, ncol=2)
        if d_idx < n_dims - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Time step $t$")
    fig.suptitle(
        f"Trajectory #{trajectory_idx} — {cfg.experiment_name}",
        y=1.02
    )
    fig.tight_layout()

    if save:
        out_dir = FIGURES_DIR / cfg.experiment_name
        _save_figure(fig, out_dir / "trajectory_comparison")

    return fig


# ---------------------------------------------------------------------------
# Batch figure generation
# ---------------------------------------------------------------------------


def generate_all_figures(
    cfg: CRNConfig,
    result: TrainingResult,
    report: EvaluationReport,
    metric_ablation: Optional[Dict[str, object]] = None,
    solver_ablation: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Generate and save all publication figures in one call.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    result:
        Training result from :func:`~train.train`.
    report:
        Evaluation report from :func:`~evaluate.evaluate`.
    metric_ablation:
        Optional metric ablation results (skipped if None).
    solver_ablation:
        Optional solver ablation results (skipped if None).
    verbose:
        Print each saved figure path.

    Returns
    -------
    list[Path]
        List of saved figure base paths (without extensions).
    """
    out_dir = FIGURES_DIR / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    def _make_and_save(func, *args, name: str, **kwargs):
        try:
            fig = func(*args, **kwargs, save=True)
            plt.close(fig)
            saved.append(out_dir / name)
            if verbose:
                print(f"  Saved: {out_dir / name}.{{pdf,png}}")
        except Exception as exc:
            if verbose:
                print(f"  WARNING: Could not generate {name}: {exc}")

    # Loss curves
    _make_and_save(plot_training_loss, result, cfg, name="training_loss")
    _make_and_save(plot_validation_loss, result, cfg, name="validation_loss")
    _make_and_save(plot_loss_curves, result, cfg, name="loss_curves")

    # Condition number over training
    if result.epoch_metrics:
        _make_and_save(plot_condition_number, result.epoch_metrics, cfg,
                       name="condition_number")

    # Convergence curves
    if report.lyapunov_exponents_per_trajectory:
        exponents = [[e] for e in report.lyapunov_exponents_per_trajectory]
        _make_and_save(plot_convergence_curves, exponents, cfg, name="convergence_curves")

    # Eigenvalue spectrum
    _make_and_save(plot_eigenvalue_spectrum, report, cfg, name="eigenvalue_spectrum")

    # Ablation figures
    if metric_ablation is not None and len(metric_ablation) > 0:
        _make_and_save(plot_metric_ablation, metric_ablation, cfg, name="ablation_metric")

    if solver_ablation is not None and len(solver_ablation) > 0:
        _make_and_save(plot_solver_ablation, solver_ablation, cfg, name="ablation_solver")

        # Runtime comparison from solver ablation data
        try:
            from ablation import AblationGroupResult
            times_dict: Dict[str, List[float]] = {}
            for name_k, gr in solver_ablation.items():
                if isinstance(gr, AblationGroupResult):
                    times_dict[name_k] = [r.mean_forward_ms for r in gr.seed_results]
                else:
                    times_dict[name_k] = [getattr(gr, "mean_forward_ms", 0.0)]
            if times_dict:
                _make_and_save(plot_runtime_comparison, times_dict, cfg,
                               name="runtime_comparison")
        except Exception:
            pass

    return saved
