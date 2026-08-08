"""
evaluate.py
===========
Evaluation suite for Convex Reasoning Networks.

Provides a comprehensive set of metrics that characterise model quality,
solver behaviour, and geometric properties of the learned metric:

+-----------------------------+------------------------------------------+
| Metric                      | Description                              |
+=============================+==========================================+
| Trajectory loss             | MSE over held-out test trajectories      |
| Convergence rate            | Spectral-radius-based Lyapunov exponent  |
| Execution time              | Per-step wall-clock time (CPU + GPU)     |
| Memory usage                | Peak resident memory during inference    |
| Condition number κ(M)       | λ_max(M) / λ_min(M)                     |
| Eigenvalue statistics       | Distribution of M's eigenvalues          |
| Success rate                | Fraction of trajectories that converge  |
+-----------------------------+------------------------------------------+

The primary entry point is :func:`evaluate`, which returns an
:class:`EvaluationReport` dataclass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from config import CRNConfig, RESULTS_DIR
from crn import CRN
from dataset import build_dataloaders
from train import trajectory_loss, CheckpointManager, build_crn
from utils import get_device


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------


@dataclass
class SolverStats:
    """Per-solver aggregated statistics from an evaluation run."""

    mean_n_iter: float
    """Mean number of iterations across all projections."""

    std_n_iter: float
    """Standard deviation of iteration counts."""

    mean_residual: float
    """Mean final primal residual."""

    convergence_rate: float
    """Fraction of solve calls that converged within tolerance."""

    mean_time_ms: float
    """Mean wall-clock time per projection (milliseconds)."""

    std_time_ms: float
    """Standard deviation of per-projection time."""


@dataclass
class MetricStats:
    """Statistics describing the learned SPD metric M."""

    condition_number: float
    """κ(M) = λ_max / λ_min."""

    log_condition_number: float
    """log κ(M) (more informative for large condition numbers)."""

    eigenvalues: List[float]
    """All eigenvalues of M in ascending order."""

    min_eigenvalue: float
    """λ_min(M) — lower bound on SPD-ness."""

    max_eigenvalue: float
    """λ_max(M)."""

    mean_eigenvalue: float
    """Mean eigenvalue."""

    std_eigenvalue: float
    """Standard deviation of eigenvalues."""

    spectral_norm_A: float
    """‖A‖_2 — must be < contraction_factor."""


@dataclass
class ConvergenceStats:
    """Lyapunov-based convergence statistics over auto-regressive rollouts."""

    lyapunov_exponents: List[float]
    """Per-trajectory estimated Lyapunov exponents."""

    mean_lyapunov: float
    """Mean Lyapunov exponent (negative → contraction)."""

    convergence_radius: float
    """Estimated radius of the invariant set."""

    success_rate: float
    """Fraction of trajectories that reach a fixed point."""

    mean_steps_to_convergence: float
    """Average steps to reach the fixed point (NaN if not converged)."""


@dataclass
class TimingStats:
    """Wall-clock and memory profiling results."""

    mean_forward_ms: float
    """Mean per-batch forward pass time (milliseconds)."""

    std_forward_ms: float
    """Standard deviation of forward pass times."""

    mean_step_ms: float
    """Mean per-time-step time (milliseconds)."""

    peak_memory_mb: float
    """Peak resident memory during evaluation (MB); 0 if profiling disabled."""

    n_repeats: int
    """Number of timed repetitions."""


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a trained CRN.

    Produced by :func:`evaluate` and serialisable to JSON via
    :meth:`to_dict`.
    """

    experiment_name: str
    test_loss: float
    val_loss: float

    solver_stats: SolverStats
    metric_stats: MetricStats
    convergence_stats: ConvergenceStats
    timing_stats: TimingStats

    n_test_trajectories: int
    n_val_trajectories: int
    n_parameters: int

    # Optional per-trajectory data for plotting
    test_losses_per_trajectory: List[float] = field(default_factory=list)
    lyapunov_exponents_per_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(i) for i in obj]
            elif isinstance(obj, float):
                return obj
            else:
                return obj

        return {
            "experiment_name": self.experiment_name,
            "test_loss": self.test_loss,
            "val_loss": self.val_loss,
            "n_test_trajectories": self.n_test_trajectories,
            "n_val_trajectories": self.n_val_trajectories,
            "n_parameters": self.n_parameters,
            "solver_stats": dataclasses.asdict(self.solver_stats),
            "metric_stats": dataclasses.asdict(self.metric_stats),
            "convergence_stats": dataclasses.asdict(self.convergence_stats),
            "timing_stats": dataclasses.asdict(self.timing_stats),
            "test_losses_per_trajectory": self.test_losses_per_trajectory,
            "lyapunov_exponents_per_trajectory": self.lyapunov_exponents_per_trajectory,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Persist the report to a JSON file.

        Parameters
        ----------
        path:
            Destination path.  Defaults to
            ``results/<experiment_name>/evaluation_report.json``.
        """
        if path is None:
            out_dir = RESULTS_DIR / self.experiment_name
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "evaluation_report.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    def summary_table(self) -> str:
        """
        Return a formatted ASCII table of the most important metrics.

        Suitable for printing to the terminal or embedding in log files.
        """
        w = 40
        lines = [
            "=" * w,
            f"  Evaluation Report: {self.experiment_name}",
            "=" * w,
            f"  Test Loss             : {self.test_loss:.6f}",
            f"  Val  Loss             : {self.val_loss:.6f}",
            f"  N Parameters          : {self.n_parameters:,}",
            f"  N Test Trajectories   : {self.n_test_trajectories}",
            "-" * w,
            "  Metric (SPD)",
            f"    Condition number κ  : {self.metric_stats.condition_number:.4f}",
            f"    log κ(M)            : {self.metric_stats.log_condition_number:.4f}",
            f"    λ_min               : {self.metric_stats.min_eigenvalue:.6f}",
            f"    λ_max               : {self.metric_stats.max_eigenvalue:.6f}",
            f"    ‖A‖₂                : {self.metric_stats.spectral_norm_A:.6f}",
            "-" * w,
            "  Solver",
            f"    Mean iterations     : {self.solver_stats.mean_n_iter:.2f}",
            f"    Convergence rate    : {self.solver_stats.convergence_rate:.4f}",
            f"    Mean time (ms)      : {self.solver_stats.mean_time_ms:.4f}",
            "-" * w,
            "  Convergence",
            f"    Mean Lyapunov exp.  : {self.convergence_stats.mean_lyapunov:.6f}",
            f"    Success rate        : {self.convergence_stats.success_rate:.4f}",
            f"    Mean steps → FP     : {self.convergence_stats.mean_steps_to_convergence:.1f}",
            "-" * w,
            "  Timing",
            f"    Mean fwd pass (ms)  : {self.timing_stats.mean_forward_ms:.4f}",
            f"    Mean/step  (ms)     : {self.timing_stats.mean_step_ms:.4f}",
            f"    Peak memory (MB)    : {self.timing_stats.peak_memory_mb:.2f}",
            "=" * w,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_trajectory_loss(
    model: CRN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[float]]:
    """
    Compute mean trajectory MSE loss over a DataLoader.

    Parameters
    ----------
    model:
        Trained CRN model.
    loader:
        DataLoader (typically test or val split).
    device:
        Compute device.

    Returns
    -------
    tuple
        ``(mean_loss, per_trajectory_losses)``
    """
    model.eval()
    all_losses: List[float] = []

    for batch in loader:
        states, inputs, contexts = batch
        states = states.to(device)
        inputs = inputs.to(device)

        x0 = states[:, 0, :]
        predicted, _ = model(x0, inputs)

        per_sample = trajectory_loss(predicted, states, reduction="none")  # (B, T)
        per_traj = per_sample.mean(dim=-1).tolist()                        # list of B floats
        all_losses.extend(per_traj)

    mean_loss = float(sum(all_losses) / max(len(all_losses), 1))
    return mean_loss, all_losses


@torch.no_grad()
def compute_convergence_stats(
    model: CRN,
    loader: DataLoader,
    device: torch.device,
    n_rollout_steps: int = 64,
) -> ConvergenceStats:
    """
    Estimate convergence statistics via long-horizon auto-regressive rollouts.

    For each trajectory, the model is run for ``n_rollout_steps`` steps
    beyond the observed sequence.  Convergence is declared when the
    step-to-step state change falls below a threshold.

    The Lyapunov exponent is estimated as::

        λ ≈ (1 / T) Σ_{t=0}^{T-1} log ‖x_{t+1} - x_t‖ / ‖x_t - x_{t-1}‖

    Parameters
    ----------
    model:
        Trained CRN model.
    loader:
        DataLoader.
    device:
        Compute device.
    n_rollout_steps:
        Number of auto-regressive rollout steps.

    Returns
    -------
    ConvergenceStats
    """
    model.eval()
    convergence_tol = 1e-4
    lyapunov_exponents: List[float] = []
    steps_to_conv: List[float] = []
    n_converged = 0
    n_total = 0

    for batch in loader:
        states, inputs, contexts = batch
        states = states.to(device)
        inputs = inputs.to(device)
        batch_size = states.shape[0]
        T_obs = inputs.shape[1]
        input_dim = inputs.shape[2]

        x0 = states[:, 0, :]

        # Pad inputs with zeros for rollout beyond observed length
        zero_pad = torch.zeros(batch_size, n_rollout_steps, input_dim,
                               dtype=inputs.dtype, device=device)
        extended_inputs = torch.cat([inputs, zero_pad], dim=1)

        # Full rollout
        traj = model.rollout(x0, extended_inputs)  # (B, T_obs+n_rollout+1, d)

        total_steps = T_obs + n_rollout_steps

        for b in range(batch_size):
            x_seq = traj[b]                            # (total_steps+1, d)
            diffs = (x_seq[1:] - x_seq[:-1]).norm(dim=-1)  # (total_steps,)

            # Lyapunov exponent estimate
            diffs_clamped = diffs.clamp(min=1e-12)
            ratios = diffs_clamped[1:] / diffs_clamped[:-1]
            exp = ratios.log().mean().item()
            lyapunov_exponents.append(exp)

            # Steps to convergence
            conv_step = total_steps  # default: did not converge
            for t_idx in range(total_steps):
                if diffs[t_idx].item() < convergence_tol:
                    conv_step = t_idx
                    n_converged += 1
                    break
            steps_to_conv.append(float(conv_step))
            n_total += 1

        break  # Only process first batch for speed (diagnostic use)

    if not lyapunov_exponents:
        lyapunov_exponents = [0.0]
        steps_to_conv = [float(n_rollout_steps)]
        n_total = 1

    mean_lyapunov = float(sum(lyapunov_exponents) / len(lyapunov_exponents))
    success_rate = n_converged / max(n_total, 1)
    mean_steps = float(sum(steps_to_conv) / max(len(steps_to_conv), 1))

    # Convergence radius: std of final states
    # Use last batch's final states as a proxy
    final_states = traj[:, -1, :]                                       # (B, d)
    centroid = final_states.mean(dim=0)
    radius = (final_states - centroid).norm(dim=-1).mean().item()

    return ConvergenceStats(
        lyapunov_exponents=lyapunov_exponents,
        mean_lyapunov=mean_lyapunov,
        convergence_radius=float(radius),
        success_rate=float(success_rate),
        mean_steps_to_convergence=mean_steps,
    )


def compute_metric_stats(model: CRN) -> MetricStats:
    """
    Compute spectral and geometric statistics of the learned metric M.

    Parameters
    ----------
    model:
        Trained CRN model.

    Returns
    -------
    MetricStats
    """
    with torch.no_grad():
        M = model.cell.metric.matrix()
        eigs = torch.linalg.eigvalsh(M)
        eigs_list = eigs.cpu().tolist()

        lam_min = float(eigs.min().item())
        lam_max = float(eigs.max().item())
        lam_mean = float(eigs.mean().item())
        lam_std = float(eigs.std().item())
        kappa = lam_max / max(lam_min, 1e-12)
        log_kappa = float(torch.log(torch.tensor(kappa)).item())

        sigma_A = float(model.spectral_norm_A().item())

    return MetricStats(
        condition_number=float(kappa),
        log_condition_number=log_kappa,
        eigenvalues=eigs_list,
        min_eigenvalue=lam_min,
        max_eigenvalue=lam_max,
        mean_eigenvalue=lam_mean,
        std_eigenvalue=lam_std,
        spectral_norm_A=sigma_A,
    )


@torch.no_grad()
def compute_solver_stats(
    model: CRN,
    loader: DataLoader,
    device: torch.device,
) -> SolverStats:
    """
    Collect solver diagnostics (iterations, residuals, convergence rate)
    over the entire DataLoader.

    Parameters
    ----------
    model:
        Trained CRN model.
    loader:
        DataLoader.
    device:
        Compute device.

    Returns
    -------
    SolverStats
    """
    model.eval()
    all_n_iters: List[float] = []
    all_residuals: List[float] = []
    all_times: List[float] = []
    n_converged = 0
    n_total = 0

    for batch in loader:
        states, inputs, contexts = batch
        states = states.to(device)
        inputs = inputs.to(device)

        x0 = states[:, 0, :]
        _, solver_results = model(x0, inputs)

        for sr in solver_results:
            all_n_iters.append(float(sr.n_iter))
            all_residuals.append(float(sr.residual))
            all_times.append(float(sr.solve_time_ms))
            if sr.converged:
                n_converged += 1
            n_total += 1

        break  # One batch is sufficient for diagnostics

    if not all_n_iters:
        all_n_iters = [0.0]
        all_residuals = [0.0]
        all_times = [0.0]
        n_total = 1

    import statistics as _stats
    mean_n_iter = _stats.mean(all_n_iters)
    std_n_iter = _stats.stdev(all_n_iters) if len(all_n_iters) > 1 else 0.0
    mean_residual = _stats.mean(all_residuals)
    mean_time = _stats.mean(all_times)
    std_time = _stats.stdev(all_times) if len(all_times) > 1 else 0.0
    conv_rate = n_converged / max(n_total, 1)

    return SolverStats(
        mean_n_iter=mean_n_iter,
        std_n_iter=std_n_iter,
        mean_residual=mean_residual,
        convergence_rate=conv_rate,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
    )


def compute_timing_stats(
    model: CRN,
    loader: DataLoader,
    device: torch.device,
    n_repeats: int = 10,
    profile_memory: bool = True,
) -> TimingStats:
    """
    Measure inference speed and peak memory usage.

    Parameters
    ----------
    model:
        Trained CRN model.
    loader:
        DataLoader (one batch is used for timing).
    device:
        Compute device.
    n_repeats:
        Number of timed forward passes.
    profile_memory:
        If True and CUDA is available, use
        :func:`torch.cuda.max_memory_allocated`.

    Returns
    -------
    TimingStats
    """
    model.eval()

    # Get one batch
    batch = next(iter(loader))
    states, inputs, contexts = batch
    states = states.to(device)
    inputs = inputs.to(device)
    x0 = states[:, 0, :]
    T = inputs.shape[1]

    # Warm-up
    with torch.no_grad():
        model(x0, inputs)

    # Reset memory counter
    peak_memory_mb = 0.0
    if profile_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    forward_times: List[float] = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(x0, inputs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            forward_times.append(elapsed_ms)

    if profile_memory and torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_bytes / (1024 ** 2)

    import statistics as _stats
    mean_fwd = _stats.mean(forward_times)
    std_fwd = _stats.stdev(forward_times) if len(forward_times) > 1 else 0.0
    mean_step_ms = mean_fwd / max(T, 1)

    return TimingStats(
        mean_forward_ms=mean_fwd,
        std_forward_ms=std_fwd,
        mean_step_ms=mean_step_ms,
        peak_memory_mb=peak_memory_mb,
        n_repeats=n_repeats,
    )


# ---------------------------------------------------------------------------
# Primary evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    cfg: CRNConfig,
    model: Optional[CRN] = None,
    checkpoint_path: Optional[Path] = None,
) -> EvaluationReport:
    """
    Run the complete evaluation suite on a trained CRN.

    Either ``model`` (an already-loaded CRN) or ``checkpoint_path`` must be
    provided.  If ``checkpoint_path`` is given, the model is loaded from disk.

    Steps:

    1. Build test and validation DataLoaders.
    2. Compute trajectory loss on test set.
    3. Compute trajectory loss on validation set.
    4. Compute solver statistics.
    5. Compute metric spectral statistics.
    6. Compute convergence statistics.
    7. Measure timing and memory.
    8. Return the full :class:`EvaluationReport`.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    model:
        Pre-loaded CRN model (optional).
    checkpoint_path:
        Path to a checkpoint file (optional; used if ``model`` is None).

    Returns
    -------
    EvaluationReport
    """
    device = get_device(cfg.train.device)

    # Build data loaders
    _, val_loader, test_loader = build_dataloaders(cfg)

    # Load or use model
    if model is None:
        if checkpoint_path is None:
            checkpoint_path = cfg.checkpoint_path("best")
        model = build_crn(cfg).to(device)
        ckpt_mgr = CheckpointManager(cfg)
        ckpt_mgr.load(Path(checkpoint_path), model, device=device)

    model = model.to(device)
    model.eval()

    # 1. Test loss
    test_loss, test_losses_per_traj = compute_trajectory_loss(model, test_loader, device)

    # 2. Val loss
    val_loss, _ = compute_trajectory_loss(model, val_loader, device)

    # 3. Solver stats
    solver_stats = compute_solver_stats(model, test_loader, device)

    # 4. Metric stats
    metric_stats = compute_metric_stats(model)

    # 5. Convergence stats
    convergence_stats = compute_convergence_stats(
        model, test_loader, device,
        n_rollout_steps=cfg.eval.n_rollout_steps,
    )

    # 6. Timing
    timing_stats = compute_timing_stats(
        model, test_loader, device,
        n_repeats=cfg.eval.timing_repeats,
        profile_memory=cfg.eval.memory_profiling,
    )

    # Count dataset sizes
    n_test = sum(1 for _ in test_loader.dataset)  # type: ignore[arg-type]
    n_val = sum(1 for _ in val_loader.dataset)    # type: ignore[arg-type]

    return EvaluationReport(
        experiment_name=cfg.experiment_name,
        test_loss=test_loss,
        val_loss=val_loss,
        solver_stats=solver_stats,
        metric_stats=metric_stats,
        convergence_stats=convergence_stats,
        timing_stats=timing_stats,
        n_test_trajectories=n_test,
        n_val_trajectories=n_val,
        n_parameters=model.n_parameters,
        test_losses_per_trajectory=test_losses_per_traj,
        lyapunov_exponents_per_trajectory=convergence_stats.lyapunov_exponents,
    )


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------


def compare_models(
    reports: Dict[str, EvaluationReport],
) -> str:
    """
    Produce a formatted ASCII comparison table from multiple evaluation reports.

    Parameters
    ----------
    reports:
        Dictionary mapping model/configuration names to their evaluation reports.

    Returns
    -------
    str
        Multi-line ASCII table string.
    """
    col_w = 16
    name_w = 22

    header = f"{'Model':<{name_w}} {'Test Loss':>{col_w}} {'Val Loss':>{col_w}} {'κ(M)':>{col_w}} {'‖A‖₂':>{col_w}} {'Succ. Rate':>{col_w}} {'Fwd ms':>{col_w}}"
    sep = "-" * len(header)

    rows = [sep, header, sep]
    for name, r in reports.items():
        row = (
            f"{name:<{name_w}} "
            f"{r.test_loss:>{col_w}.6f} "
            f"{r.val_loss:>{col_w}.6f} "
            f"{r.metric_stats.condition_number:>{col_w}.4f} "
            f"{r.metric_stats.spectral_norm_A:>{col_w}.6f} "
            f"{r.convergence_stats.success_rate:>{col_w}.4f} "
            f"{r.timing_stats.mean_forward_ms:>{col_w}.4f}"
        )
        rows.append(row)
    rows.append(sep)
    return "\n".join(rows)

