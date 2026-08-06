
"""
model.py
========
Self-contained Convex Reasoning Network (CRN) model.

Implements the full CRN architecture as a single, importable
:class:`ConvexReasoningNetwork` module.  The state-update rule is::

    x_{t+1} = Prox_C^M((I + A) x_t + B g_t)

where:

* ``M``  is a learnable SPD metric parameterised by a Cholesky factor
* ``A``  is a contractive residual map  (‖I + A‖₂ < contraction_factor)
* ``B``  is a learnable input-to-state matrix
* ``C``  is the convex hull of ``K`` learnable context prototype vectors

The model exposes a clean public API:

* :meth:`ConvexReasoningNetwork.forward` — unroll the recurrence for T steps
* :meth:`ConvexReasoningNetwork.rollout` — return the full state trajectory
* :meth:`ConvexReasoningNetwork.fixed_point` — run to a fixed point for fixed g
* :meth:`ConvexReasoningNetwork.save_checkpoint` — persist weights + metadata
* :meth:`ConvexReasoningNetwork.load_checkpoint` — restore weights + metadata
* :meth:`ConvexReasoningNetwork.reset_parameters` — re-initialise all weights
* :meth:`ConvexReasoningNetwork.extra_repr` — human-readable summary string
* :meth:`ConvexReasoningNetwork.parameter_summary` — per-component counts

Exports
-------
* :class:`ConvexReasoningNetwork`
* :func:`build_model`
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from config import CHECKPOINTS_DIR, CRNConfig, ModelConfig
from geometry import ConvexHullContext, project_onto_simplex
from utils import (
    count_parameters,
    get_logger,
    model_size_mb,
    safe_cholesky,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal SPD metric
# ---------------------------------------------------------------------------


class _SPDMetric(nn.Module):
    """
    Learnable symmetric positive-definite (SPD) metric.

    The metric matrix is parameterised as::

        M = L L^T + eps * I

    where ``L`` is a learnable lower-triangular matrix with unconstrained
    entries and ``eps > 0`` is a fixed regularisation constant.  This
    guarantees that ``M`` is always SPD regardless of the values of ``L``,
    making it safe to optimise with unconstrained gradient descent.

    Parameters
    ----------
    dim:
        Dimension ``d`` of the ambient state space.
    eps:
        Regularisation constant ``eps`` (default ``1e-4``).
    """

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Lower-triangular parameter matrix (unconstrained)
        self._L_raw: nn.Parameter = nn.Parameter(torch.empty(dim, dim))
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialise L_raw so that M starts near the identity."""
        nn.init.eye_(self._L_raw)
        with torch.no_grad():
            # Keep only the lower triangle; zero the strict upper part
            mask = torch.tril(torch.ones_like(self._L_raw))
            self._L_raw.data.mul_(mask)

    def _build_L(self) -> Tensor:
        """Enforce lower-triangular structure and positive diagonal."""
        L = torch.tril(self._L_raw)
        # Positive diagonal via softplus so L stays invertible
        diag = torch.nn.functional.softplus(L.diagonal()) + 1e-6
        L = L.clone()
        L.diagonal().copy_(diag)
        return L

    def matrix(self) -> Tensor:
        """
        Return the full SPD metric matrix ``M = L L^T + eps * I``.

        Returns
        -------
        Tensor
            Shape ``(dim, dim)``.
        """
        L = self._build_L()
        eye = torch.eye(self.dim, dtype=L.dtype, device=L.device)
        return L @ L.t() + self.eps * eye

    def cholesky(self) -> Tensor:
        """
        Return the true Cholesky factor of ``M``.

        Because ``M = L_raw L_raw^T + eps I`` the *stored* ``L_raw`` is not
        the exact Cholesky factor of ``M`` (the ``eps I`` term shifts the
        eigenvalues).  This method recomputes the factorisation of the full
        ``M`` to honour the contract ``chol @ chol.T == M``.

        Returns
        -------
        Tensor
            Lower-triangular matrix of shape ``(dim, dim)``.
        """
        return torch.linalg.cholesky(self.matrix())

    def apply(self, x: Tensor) -> Tensor:
        """
        Compute the metric-weighted product ``M x``.

        Parameters
        ----------
        x:
            Tensor of shape ``(*, dim)``.

        Returns
        -------
        Tensor
            ``M x``, same shape as ``x``.
        """
        M = self.matrix()
        shape = x.shape
        x2d = x.reshape(-1, self.dim)
        result = (M @ x2d.t()).t()
        return result.reshape(shape)

    def solve(self, x: Tensor) -> Tensor:
        """
        Solve ``M z = x`` (i.e. compute ``z = M^{-1} x``).

        Uses the Cholesky factorisation of ``M`` for numerical stability.

        Parameters
        ----------
        x:
            Right-hand side tensor of shape ``(*, dim)``.

        Returns
        -------
        Tensor
            Solution ``z`` of the same shape.
        """
        chol = self.cholesky()
        shape = x.shape
        x2d = x.reshape(-1, self.dim).t()
        y = torch.linalg.solve_triangular(chol, x2d, upper=False)
        z = torch.linalg.solve_triangular(chol.t(), y, upper=True)
        return z.t().reshape(shape)

    def eigenvalues(self) -> Tensor:
        """
        Return the eigenvalues of ``M`` in ascending order.

        Returns
        -------
        Tensor
            Shape ``(dim,)``.
        """
        return torch.linalg.eigvalsh(self.matrix())

    def condition_number(self) -> Tensor:
        """
        Return ``lambda_max(M) / lambda_min(M)`` (condition number of M).

        Returns
        -------
        Tensor
            Scalar condition number.
        """
        eigs = self.eigenvalues()
        return eigs.max() / eigs.min().clamp(min=1e-12)

    def log_determinant(self) -> Tensor:
        """
        Compute ``log |M|`` via the Cholesky factor (numerically stable).

        Returns
        -------
        Tensor
            Scalar log-determinant.
        """
        chol = self.cholesky()
        return 2.0 * chol.diagonal().log().sum()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Euclidean (identity) metric — ablation baseline
# ---------------------------------------------------------------------------


class _EuclideanMetric(nn.Module):
    """
    Fixed Euclidean (identity) metric: ``M = I``.

    This is a parameter-free baseline used in ablation studies to isolate
    the contribution of the learnable SPD metric.

    Parameters
    ----------
    dim:
        Dimension of the ambient space.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.eps = 0.0

    def matrix(self) -> Tensor:
        """Return the identity matrix of shape ``(dim, dim)``."""
        return torch.eye(self.dim)

    def cholesky(self) -> Tensor:
        """Return the identity matrix (its own Cholesky factor)."""
        return torch.eye(self.dim)

    def apply(self, x: Tensor) -> Tensor:
        """Identity map: returns ``x`` unchanged."""
        return x

    def solve(self, x: Tensor) -> Tensor:
        """Identity inverse: returns ``x`` unchanged."""
        return x

    def eigenvalues(self) -> Tensor:
        """Return a vector of ones."""
        return torch.ones(self.dim)

    def condition_number(self) -> Tensor:
        """Return 1.0 (identity is perfectly conditioned)."""
        return torch.tensor(1.0)

    def log_determinant(self) -> Tensor:
        """Return 0.0 (log-determinant of identity is zero)."""
        return torch.tensor(0.0)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ---------------------------------------------------------------------------
# Proximal solver (analytic + PGD fallback)
# ---------------------------------------------------------------------------


def _prox_analytic(
    v: Tensor,
    convex_set: ConvexHullContext,
    metric: _SPDMetric | _EuclideanMetric,
) -> Tensor:
    """
    Compute ``Prox_C^M(v)`` via an analytic active-set QP.

    Solves::

        x* = argmin_{x ∈ C}  ½ ‖x - v‖_M²
           = argmin_{α ∈ Δ^{K-1}}  ½ α^T (C M^{-1} C^T) α - (v^T M^{-1} C^T) α

    using projected gradient descent on the simplex (30 iterations).

    Parameters
    ----------
    v:
        Pre-projection input, shape ``(batch, state_dim)``.
    convex_set:
        The convex hull context ``C`` (prototype matrix: ``(K, state_dim)``).
    metric:
        The SPD metric ``M``.

    Returns
    -------
    Tensor
        Proximal solution of shape ``(batch, state_dim)``.
    """
    C = convex_set.prototypes          # (K, state_dim)
    K = C.shape[0]

    # Solve in the M^{-1}-weighted inner-product space.
    # Compute M^{-1} C^T: (state_dim, K)  →  then Q = C M^{-1} C^T: (K, K)
    MC_t = metric.solve(C.t())         # (state_dim, K)
    Q = C @ MC_t                       # (K, K)  (Gram matrix under M^{-1})

    # Linear term: b[b, k] = v[b] @ M^{-1} c_k
    Mv_t = metric.solve(v.t())         # (state_dim, batch)
    b = (C @ Mv_t).t()                 # (batch, K)

    # Warm-start: project correlation scores onto simplex
    alpha = project_onto_simplex(b)    # (batch, K)

    # Step size: 1 / lambda_max(Q)
    with torch.no_grad():
        lam_max = torch.linalg.eigvalsh(Q).max().clamp(min=1e-8)
        step = 1.0 / lam_max.item()

    for _ in range(50):
        grad = alpha @ Q.t() - b       # (batch, K)
        alpha_new = project_onto_simplex(alpha - step * grad)
        delta = (alpha_new - alpha).norm()
        alpha = alpha_new
        if delta.item() < 1e-7:
            break

    return alpha @ C                   # (batch, state_dim)


def _prox_pgd(
    v: Tensor,
    convex_set: ConvexHullContext,
    metric: _SPDMetric | _EuclideanMetric,
    step_size: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tensor:
    """
    Compute ``Prox_C^M(v)`` via projected gradient descent (PGD).

    Falls back to PGD when the analytic approach is numerically difficult.

    Parameters
    ----------
    v:
        Pre-projection input, shape ``(batch, state_dim)``.
    convex_set:
        The convex hull context set C.
    metric:
        The SPD metric M.
    step_size:
        Gradient step size η (should satisfy η < 2 / λ_max(M)).
    max_iter:
        Maximum number of PGD iterations.
    tol:
        Convergence tolerance on successive iterate differences.

    Returns
    -------
    Tensor
        Proximal solution of shape ``(batch, state_dim)``.
    """
    z = convex_set.project(v)

    for _ in range(max_iter):
        # Gradient of ½ ‖z - v‖_M² = M(z - v)
        grad = metric.apply(z - v)
        z_new = convex_set.project(z - step_size * grad)
        if (z_new - z).norm(dim=-1).max().item() < tol:
            z = z_new
            break
        z = z_new

    return z


# ---------------------------------------------------------------------------
# Contractivity utility
# ---------------------------------------------------------------------------


def _make_contractive(W: Tensor, factor: float = 0.9) -> Tensor:
    """
    Re-parameterise ``W`` so that ``‖W‖_2 ≤ factor``.

    Parameters
    ----------
    W:
        Raw weight matrix of shape ``(d, d)``.
    factor:
        Target spectral-norm upper bound (< 1 for contractivity).

    Returns
    -------
    Tensor
        Normalised matrix with the same shape.
    """
    sigma = torch.linalg.matrix_norm(W, ord=2)
    scale = sigma.clamp(min=1.0)
    return factor * W / scale


# ---------------------------------------------------------------------------
# CRN cell — single time step
# ---------------------------------------------------------------------------


class _CRNCell(nn.Module):
    """
    Single-step update of the Convex Reasoning Network.

    Computes::

        v_t     = (I + A) x_t + B g_t
        x_{t+1} = Prox_C^M(v_t)

    where ``(I + A)`` is a contractive linear map whose full spectral norm is
    bounded by ``contraction_factor``.

    Parameters
    ----------
    state_dim:
        Dimension of the state vector.
    input_dim:
        Dimension of the exogenous input.
    n_context_vectors:
        Number of prototype atoms in the convex hull ``C``.
    metric:
        Pre-constructed metric module.
    convex_set:
        Pre-constructed convex-hull context set.
    contraction_factor:
        Upper bound on ``‖I + A‖_2``.
    solver:
        Which proximal solver to use: ``'analytic'`` or ``'pgd'``.
    solver_step_size:
        PGD step size (only used when ``solver='pgd'``).
    solver_max_iter:
        Maximum solver iterations (only used when ``solver='pgd'``).
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        n_context_vectors: int,
        metric: _SPDMetric | _EuclideanMetric,
        convex_set: ConvexHullContext,
        contraction_factor: float = 0.9,
        solver: str = "analytic",
        solver_step_size: float = 0.1,
        solver_max_iter: int = 50,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_context_vectors = n_context_vectors
        self.contraction_factor = contraction_factor
        self.solver = solver
        self.solver_step_size = solver_step_size
        self.solver_max_iter = solver_max_iter

        self.metric = metric
        self.convex_set = convex_set

        # Learnable parameters
        self._A_raw: nn.Parameter = nn.Parameter(
            torch.empty(state_dim, state_dim)
        )
        self.B: nn.Parameter = nn.Parameter(
            torch.empty(state_dim, input_dim)
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Kaiming-uniform initialisation, scaled small for stable early training."""
        nn.init.kaiming_uniform_(self._A_raw, a=math.sqrt(5))
        with torch.no_grad():
            self._A_raw.data.mul_(0.1)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    @property
    def A(self) -> Tensor:
        """
        Contractive residual matrix.

        The *full* linear map ``T = I + A`` satisfies ``‖T‖_2 ≤
        contraction_factor < 1`` by construction: we apply spectral
        normalisation to ``I + A_raw`` and recover ``A = T - I``.  This
        guarantees fixed-point convergence for the proximal iteration.
        """
        eye = torch.eye(
            self._A_raw.shape[0],
            device=self._A_raw.device,
            dtype=self._A_raw.dtype,
        )
        T = _make_contractive(eye + self._A_raw, factor=self.contraction_factor)
        return T - eye

    def forward(
        self,
        x_t: Tensor,
        g_t: Tensor,
    ) -> Tensor:
        """
        Perform one CRN step.

        Parameters
        ----------
        x_t:
            Current state of shape ``(batch, state_dim)``.
        g_t:
            Exogenous input of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Next state ``x_{t+1}`` of shape ``(batch, state_dim)``.
        """
        # Pre-projection: v_t = (I + A) x_t + B g_t
        A = self.A
        v_t = x_t + x_t @ A.t() + g_t @ self.B.t()

        # Proximal projection onto C under metric M
        if self.solver == "pgd":
            return _prox_pgd(
                v_t,
                self.convex_set,
                self.metric,
                step_size=self.solver_step_size,
                max_iter=self.solver_max_iter,
            )
        return _prox_analytic(v_t, self.convex_set, self.metric)

    def energy(self, x_t: Tensor, g_t: Tensor) -> Tensor:
        """
        Lyapunov energy ``½ ‖x_{t+1} - x_t‖_M²``.

        A non-increasing energy confirms convergence toward the fixed point.

        Parameters
        ----------
        x_t:
            Current state of shape ``(batch, state_dim)``.
        g_t:
            Input of shape ``(batch, input_dim)``.

        Returns
        -------
        Tensor
            Per-sample energy of shape ``(batch,)``.
        """
        x_next = self.forward(x_t, g_t)
        diff = x_next - x_t
        Mdiff = self.metric.apply(diff)
        return 0.5 * (diff * Mdiff).sum(dim=-1)


# ---------------------------------------------------------------------------
# Full ConvexReasoningNetwork model
# ---------------------------------------------------------------------------


class ConvexReasoningNetwork(nn.Module):
    """
    Convex Reasoning Network (CRN) — full sequence model.

    Unrolls the CRN recurrence for an input sequence of length ``T``,
    returning the predicted state trajectory.

    Architecture overview
    ---------------------
    ::

        x_0  (given)
        x_1  = Prox_C^M((I + A) x_0 + B g_0)
        x_2  = Prox_C^M((I + A) x_1 + B g_1)
        ...
        x_T  = Prox_C^M((I + A) x_{T-1} + B g_{T-1})

    Learnable components
    --------------------
    * ``A_raw``  — raw weight matrix; processed by spectral normalisation to
                   ensure ``‖I + A‖_2 ≤ contraction_factor < 1``.
    * ``B``      — input-to-state matrix.
    * ``L_raw``  — lower-triangular Cholesky factor for the SPD metric
                   ``M = L L^T + eps I`` (not present for Euclidean metric).
    * ``prototypes`` — ``K`` learnable context prototype vectors defining
                       the convex hull ``C = conv({c_1, …, c_K})``.

    Parameters
    ----------
    cfg:
        Experiment configuration.  All architectural hyper-parameters are
        read from ``cfg.model``; solver hyper-parameters from ``cfg.solver``.

    Notes
    -----
    * :meth:`save_checkpoint` / :meth:`load_checkpoint` handle persistence.
    * :meth:`reset_parameters` re-initialises all learnable weights.
    * :meth:`fixed_point` iterates to the fixed point for a constant input.
    """

    def __init__(self, cfg: CRNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        mcfg: ModelConfig = cfg.model

        # Metric
        if mcfg.metric_type == "spd":
            self._metric: _SPDMetric | _EuclideanMetric = _SPDMetric(
                dim=mcfg.state_dim,
                eps=mcfg.metric_eps,
            )
        elif mcfg.metric_type == "euclidean":
            self._metric = _EuclideanMetric(dim=mcfg.state_dim)
        else:
            raise ValueError(
                f"Unknown metric_type '{mcfg.metric_type}'. "
                "Expected 'spd' or 'euclidean'."
            )

        # Convex set C = conv(prototypes)
        self._convex_set = ConvexHullContext(
            state_dim=mcfg.state_dim,
            n_context_vectors=mcfg.n_context_vectors,
        )

        # Core cell
        self._cell = _CRNCell(
            state_dim=mcfg.state_dim,
            input_dim=mcfg.input_dim,
            n_context_vectors=mcfg.n_context_vectors,
            metric=self._metric,
            convex_set=self._convex_set,
            contraction_factor=mcfg.contraction_factor,
            solver=mcfg.solver,
            solver_step_size=cfg.solver.pgd_step_size,
            solver_max_iter=cfg.solver.max_iter,
        )

    # ------------------------------------------------------------------
    # Forward / rollout / fixed-point
    # ------------------------------------------------------------------

    def forward(
        self,
        x0: Tensor,
        inputs: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Unroll the CRN recurrence for ``T`` steps.

        Parameters
        ----------
        x0:
            Initial state of shape ``(batch, state_dim)``.
        inputs:
            Exogenous input sequence of shape ``(batch, T, input_dim)``.

        Returns
        -------
        tuple
            ``(states, energies)`` where:

            * ``states``   — tensor of shape ``(batch, T+1, state_dim)``
              holding ``[x_0, x_1, …, x_T]``.
            * ``energies`` — list of ``T`` per-step Lyapunov energy tensors,
              each of shape ``(batch,)``.
        """
        batch, T, _ = inputs.shape
        states: List[Tensor] = [x0]
        energies: List[Tensor] = []

        x = x0
        for t in range(T):
            g_t = inputs[:, t, :]
            energies.append(self._cell.energy(x, g_t))
            x = self._cell(x, g_t)
            states.append(x)

        return torch.stack(states, dim=1), energies

    def rollout(
        self,
        x0: Tensor,
        inputs: Tensor,
    ) -> Tensor:
        """
        Return only the state trajectory tensor.

        Equivalent to ``forward(x0, inputs)[0]`` but slightly cleaner to
        call when the energy diagnostics are not needed.

        Parameters
        ----------
        x0:
            Initial state of shape ``(batch, state_dim)``.
        inputs:
            Exogenous input sequence of shape ``(batch, T, input_dim)``.

        Returns
        -------
        Tensor
            State trajectory of shape ``(batch, T+1, state_dim)``.
        """
        states, _ = self.forward(x0, inputs)
        return states

    def fixed_point(
        self,
        g: Tensor,
        x_init: Optional[Tensor] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> Tuple[Tensor, int, bool]:
        """
        Iterate the CRN cell to a fixed point for a constant input ``g``.

        Runs the recurrence ``x_{k+1} = cell(x_k, g)`` until convergence
        (norm of successive iterates < ``tol``) or ``max_iter`` steps.

        This is useful for computing the equilibrium state corresponding to
        a constant driving input, e.g. for fixed-point stability analysis.

        Parameters
        ----------
        g:
            Constant exogenous input of shape ``(batch, input_dim)`` or
            ``(input_dim,)`` (will be unsqueezed if necessary).
        x_init:
            Initial state of shape ``(batch, state_dim)``.  Defaults to
            the zero vector when ``None``.
        max_iter:
            Maximum number of iterations.
        tol:
            Convergence tolerance on ``‖x_{k+1} - x_k‖_∞``.

        Returns
        -------
        tuple
            ``(x_star, n_iter, converged)`` where ``x_star`` has shape
            ``(batch, state_dim)``, ``n_iter`` is the number of steps taken,
            and ``converged`` is ``True`` when the tolerance was met.
        """
        state_dim = self.cfg.model.state_dim

        if g.dim() == 1:
            g = g.unsqueeze(0)
        batch = g.shape[0]

        if x_init is None:
            x = torch.zeros(batch, state_dim, dtype=g.dtype, device=g.device)
        else:
            x = x_init.clone()

        converged = False
        n_iter = 0

        with torch.no_grad():
            for n_iter in range(1, max_iter + 1):
                x_new = self._cell(x, g)
                residual = (x_new - x).abs().max().item()
                x = x_new
                if residual < tol:
                    converged = True
                    break

        return x, n_iter, converged

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Optional[Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        val_loss: float = float("inf"),
        scheduler: Optional[Any] = None,
        tag: str = "best",
    ) -> Path:
        """
        Persist model weights and training metadata to a ``.pt`` file.

        The checkpoint dictionary contains:

        * ``model_state_dict`` — :meth:`state_dict` of this module.
        * ``optimizer_state_dict`` — optimizer state (if provided).
        * ``scheduler_state_dict`` — LR-scheduler state (if provided).
        * ``epoch`` — current training epoch.
        * ``val_loss`` — validation loss at this checkpoint.
        * ``cfg`` — JSON-serialised :class:`CRNConfig`.

        Parameters
        ----------
        path:
            Destination file path.  Defaults to
            ``checkpoints/<experiment_name>/<tag>.pt``.
        optimizer:
            Optimizer whose state should be saved alongside the weights.
        epoch:
            Current epoch number (stored for resume-training support).
        val_loss:
            Validation loss at this checkpoint.
        scheduler:
            LR scheduler whose state should be saved (optional).
        tag:
            Short label appended to the default filename (e.g. ``'best'``,
            ``'epoch_0010'``).

        Returns
        -------
        Path
            Absolute path of the saved checkpoint.
        """
        if path is None:
            ckpt_dir = CHECKPOINTS_DIR / self.cfg.experiment_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"{tag}.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "cfg": self.cfg.to_json(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(payload, path)
        logger.info("Checkpoint saved → %s  (epoch %d, val_loss=%.6f)", path, epoch, val_loss)
        return path

    def load_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Restore model weights (and optionally optimizer / scheduler state).

        Parameters
        ----------
        path:
            Path to a ``.pt`` checkpoint file previously written by
            :meth:`save_checkpoint`.
        optimizer:
            Optimizer to restore state into (optional).
        scheduler:
            LR scheduler to restore state into (optional).
        device:
            Map location for the loaded tensors (defaults to current
            device of the model's first parameter, or CPU).

        Returns
        -------
        dict
            The raw checkpoint dictionary, containing at minimum:
            ``epoch``, ``val_loss``, and ``cfg``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        payload: Dict[str, Any] = torch.load(
            path, map_location=device, weights_only=False
        )
        self.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])

        logger.info(
            "Checkpoint loaded ← %s  (epoch %d, val_loss=%.6f)",
            path,
            payload.get("epoch", 0),
            payload.get("val_loss", float("inf")),
        )
        return payload

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """
        Re-initialise all learnable parameters to their default values.

        This is equivalent to constructing a fresh model with the same
        configuration.  Useful for multi-seed ablation studies where the
        same :class:`ConvexReasoningNetwork` object is re-used across runs.

        Components reset:

        * ``A_raw`` — small random Kaiming-uniform matrix (scaled by 0.1).
        * ``B`` — Kaiming-uniform matrix.
        * ``L_raw`` (SPD metric) — near-identity lower-triangular matrix.
        * ``prototypes`` (convex hull) — scaled normal random vectors.
        """
        self._cell._init_parameters()
        if isinstance(self._metric, _SPDMetric):
            self._metric._init_parameters()
        self._convex_set._init_parameters(scale=0.1)
        logger.debug("ConvexReasoningNetwork parameters reset.")

    # ------------------------------------------------------------------
    # Diagnostic properties and methods
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return count_parameters(self, trainable_only=True)

    @property
    def size_mb(self) -> float:
        """Approximate memory footprint of all parameters in megabytes."""
        return model_size_mb(self)

    def parameter_summary(self) -> Dict[str, int]:
        """
        Return a per-component breakdown of trainable parameter counts.

        Returns
        -------
        dict
            Keys: ``'A'``, ``'B'``, ``'metric'``, ``'context'``, ``'total'``.
        """
        return {
            "A": self._cell._A_raw.numel(),
            "B": self._cell.B.numel(),
            "metric": count_parameters(self._metric, trainable_only=True),
            "context": count_parameters(self._convex_set, trainable_only=True),
            "total": self.n_parameters,
        }

    def spectral_norm_A(self) -> Tensor:
        """
        Return ``‖I + A‖_2`` — the spectral norm of the *full* linear map.

        This quantity must be < ``contraction_factor`` for the fixed-point
        convergence guarantee to hold.  (Note: ``‖A‖_2 < 1`` alone does not
        imply ``‖I + A‖_2 < 1``.)

        Returns
        -------
        Tensor
            Scalar spectral norm.
        """
        with torch.no_grad():
            A = self._cell.A
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            return torch.linalg.matrix_norm(eye + A, ord=2)

    def condition_number_M(self) -> Tensor:
        """
        Return the condition number ``κ(M) = λ_max(M) / λ_min(M)``.

        Returns
        -------
        Tensor
            Scalar condition number (≥ 1).
        """
        with torch.no_grad():
            return self._metric.condition_number()

    def log_det_M(self) -> Tensor:
        """
        Return ``log |M|`` (log-determinant of the metric matrix).

        Returns
        -------
        Tensor
            Scalar log-determinant.
        """
        with torch.no_grad():
            return self._metric.log_determinant()

    def eigenvalues_M(self) -> Tensor:
        """
        Return the eigenvalues of ``M`` in ascending order.

        Returns
        -------
        Tensor
            Shape ``(state_dim,)``.
        """
        with torch.no_grad():
            return self._metric.eigenvalues()

    def extra_repr(self) -> str:
        """
        Return a compact human-readable summary of the model configuration.

        Displayed automatically by :func:`print` (via ``nn.Module.__repr__``).
        """
        mcfg = self.cfg.model
        return (
            f"state_dim={mcfg.state_dim}, "
            f"input_dim={mcfg.input_dim}, "
            f"n_context_vectors={mcfg.n_context_vectors}, "
            f"metric_type={mcfg.metric_type}, "
            f"solver={mcfg.solver}, "
            f"contraction_factor={mcfg.contraction_factor}, "
            f"n_parameters={self.n_parameters}"
        )

    def __repr__(self) -> str:
        base = super().__repr__()
        summary = (
            f"\n  [ConvexReasoningNetwork summary]\n"
            f"  Parameters : {self.n_parameters:,}\n"
            f"  Size       : {self.size_mb:.3f} MB\n"
            f"  Metric     : {self.cfg.model.metric_type.upper()}\n"
            f"  Solver     : {self.cfg.model.solver}\n"
            f"  state_dim  : {self.cfg.model.state_dim}\n"
            f"  input_dim  : {self.cfg.model.input_dim}\n"
            f"  K (atoms)  : {self.cfg.model.n_context_vectors}\n"
            f"  factor     : {self.cfg.model.contraction_factor}"
        )
        return base + summary


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(cfg: CRNConfig) -> ConvexReasoningNetwork:
    """
    Construct a :class:`ConvexReasoningNetwork` from a configuration object.

    This is the canonical entry-point used by training and evaluation scripts.
    It is equivalent to ``ConvexReasoningNetwork(cfg)`` but reads more clearly
    at call sites.

    Parameters
    ----------
    cfg:
        Experiment configuration.

    Returns
    -------
    ConvexReasoningNetwork
        Freshly initialised model on CPU.  Call ``.to(device)`` to move it.

    Example
    -------
    >>> from config import CRNConfig
    >>> model = build_model(CRNConfig())
    >>> print(model.n_parameters)
    """
    return ConvexReasoningNetwork(cfg)
