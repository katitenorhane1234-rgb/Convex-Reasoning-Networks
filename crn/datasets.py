"""
dataset.py
==========
Synthetic trajectory dataset for Convex Reasoning Networks.

A stable ground-truth linear dynamical system generates noisily-observed
state sequences.  The module exposes:

* :func:`generate_ground_truth_system`  — sample A_true, B_true, C_true
* :func:`generate_trajectories`         — roll out the system into tensors
* :func:`split_trajectories`            — partition into train / val / test splits
* :class:`TrajectoryDataset`            — ``torch.utils.data.Dataset`` wrapper
* :func:`build_dataloaders`             — single entry-point: generate + split + wrap
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from config import CRNConfig, DataConfig


# ---------------------------------------------------------------------------
# Named container for a single split
# ---------------------------------------------------------------------------


class TrajectorySplit(NamedTuple):
    """Holds the tensors for one dataset split (train, val, or test)."""

    states: Tensor
    """Shape: (N, T+1, state_dim) — state sequence including initial state x_0."""

    inputs: Tensor
    """Shape: (N, T, input_dim) — exogenous inputs g_t for t = 0 … T-1."""

    contexts: Tensor
    """Shape: (N, T, context_dim) — convex combination weights over context vectors."""


# ---------------------------------------------------------------------------
# Ground-truth system
# ---------------------------------------------------------------------------


class GroundTruthSystem(NamedTuple):
    """Stable linear system used to generate synthetic data."""

    A_true: Tensor
    """Shape: (state_dim, state_dim) — stable state-transition matrix."""

    B_true: Tensor
    """Shape: (state_dim, input_dim) — input matrix."""

    C_true: Tensor
    """Shape: (context_dim, state_dim) — context prototype matrix."""


def generate_ground_truth_system(cfg: DataConfig, seed: int) -> GroundTruthSystem:
    """
    Sample a stable ground-truth linear dynamical system.

    Stability is enforced by eigenvalue normalisation: A_true is constructed
    so that its spectral radius is strictly less than 1.

    Parameters
    ----------
    cfg:
        Data configuration (supplies dimensionalities).
    seed:
        Random seed for reproducible system generation.

    Returns
    -------
    GroundTruthSystem
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    d = cfg.state_dim
    p = cfg.input_dim
    K = cfg.context_weight_dim

    # Generate random A and normalise to ensure spectral radius < 0.95
    A_raw = torch.randn(d, d, generator=rng)
    # SVD-based normalisation: A_true = 0.9 * A_raw / σ_max(A_raw)
    U, S, Vh = torch.linalg.svd(A_raw)
    A_true = 0.9 * U @ torch.diag(S / S.max()) @ Vh

    # Verify stability
    with torch.no_grad():
        rho = torch.linalg.eigvals(A_true).abs().max()
        if rho >= 1.0:
            # Extra safety: scale down
            A_true = A_true * (0.9 / rho.real)

    # Random input matrix B
    B_true = torch.randn(d, p, generator=rng) * 0.5

    # Random context prototype matrix C_true: rows are context vectors in ℝ^d
    C_true = torch.randn(K, d, generator=rng) * 0.3

    return GroundTruthSystem(A_true=A_true, B_true=B_true, C_true=C_true)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def generate_trajectories(
    system: GroundTruthSystem,
    cfg: DataConfig,
    seed: int,
) -> TrajectorySplit:
    """
    Roll out the ground-truth system to produce noisy trajectories.

    The dynamics follow::

        x_{t+1} = A_true x_t + B_true g_t + ε_t,   ε_t ~ N(0, σ²I)

    where context weights are drawn from a Dirichlet distribution so that they
    always form valid convex combinations.

    Parameters
    ----------
    system:
        Ground-truth system matrices.
    cfg:
        Data configuration.
    seed:
        Random seed for trajectory generation (distinct from the system seed).

    Returns
    -------
    TrajectorySplit
        Tensors of shape (n_trajectories, …).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    N = cfg.n_trajectories
    T = cfg.trajectory_length
    d = cfg.state_dim
    p = cfg.input_dim
    K = cfg.context_weight_dim
    sigma = cfg.noise_std

    A = system.A_true     # (d, d)
    B = system.B_true     # (d, p)

    # Allocate tensors
    states = torch.zeros(N, T + 1, d)
    inputs = torch.randn(N, T, p, generator=rng)

    # Sample context weights from Dirichlet(1, …, 1) — uniform over simplex
    # Using the Gamma/normalise method for reproducibility
    gamma_samples = torch.empty(N, T, K).exponential_(generator=rng)
    contexts = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    # Initial states — small random values
    states[:, 0, :] = torch.randn(N, d, generator=rng) * 0.1

    # Roll out the dynamics
    noise = torch.randn(N, T, d, generator=rng) * sigma

    for t in range(T):
        x_t = states[:, t, :]                    # (N, d)
        g_t = inputs[:, t, :]                    # (N, p)
        # x_{t+1} = A x_t + B g_t + noise_t
        x_next = x_t @ A.t() + g_t @ B.t() + noise[:, t, :]
        states[:, t + 1, :] = x_next

    return TrajectorySplit(states=states, inputs=inputs, contexts=contexts)


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------


def split_trajectories(
    data: TrajectorySplit,
    cfg: DataConfig,
) -> Tuple[TrajectorySplit, TrajectorySplit, TrajectorySplit]:
    """
    Deterministically partition trajectories into train / val / test splits.

    Parameters
    ----------
    data:
        Full dataset returned by :func:`generate_trajectories`.
    cfg:
        Data configuration (supplies split fractions and seed).

    Returns
    -------
    tuple
        ``(train_split, val_split, test_split)``
    """
    N = data.states.shape[0]

    # Compute split sizes
    n_train = int(N * cfg.train_frac)
    n_val = int(N * cfg.val_frac)
    n_test = N - n_train - n_val

    # Ensure non-empty test set
    if n_test <= 0:
        n_val = max(1, n_val - 1)
        n_test = N - n_train - n_val

    # Deterministic permutation
    rng = torch.Generator()
    rng.manual_seed(cfg.seed)
    perm = torch.randperm(N, generator=rng)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def _slice(idx: Tensor) -> TrajectorySplit:
        return TrajectorySplit(
            states=data.states[idx],
            inputs=data.inputs[idx],
            contexts=data.contexts[idx],
        )

    return _slice(train_idx), _slice(val_idx), _slice(test_idx)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class TrajectoryDataset(Dataset):
    """
    A ``torch.utils.data.Dataset`` wrapping a :class:`TrajectorySplit`.

    Each item is a tuple ``(states, inputs, contexts)`` for one trajectory.

    Parameters
    ----------
    split:
        The trajectory split to wrap.
    """

    def __init__(self, split: TrajectorySplit) -> None:
        self.states = split.states
        self.inputs = split.inputs
        self.contexts = split.contexts

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.states[idx], self.inputs[idx], self.contexts[idx]

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories in this split."""
        return self.states.shape[0]

    @property
    def trajectory_length(self) -> int:
        """Number of time steps T (excluding the initial state)."""
        return self.states.shape[1] - 1

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""
        return self.states.shape[2]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: CRNConfig,
    *,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test :class:`~torch.utils.data.DataLoader` objects.

    This is the primary entry point for data loading.  It handles:

    1. Ground-truth system generation (seeded).
    2. Trajectory simulation (seeded).
    3. Deterministic train / val / test split.
    4. DataLoader construction with reproducible worker seeding.

    Parameters
    ----------
    cfg:
        Full experiment configuration.
    pin_memory:
        Whether to use pinned memory (speeds up CPU→GPU transfers).

    Returns
    -------
    tuple
        ``(train_loader, val_loader, test_loader)``
    """
    system = generate_ground_truth_system(cfg.data, seed=cfg.data.seed)
    full_data = generate_trajectories(system, cfg.data, seed=cfg.data.seed + 1)
    train_split, val_split, test_split = split_trajectories(full_data, cfg.data)

    def _seed_worker(worker_id: int) -> None:
        import random
        worker_seed = cfg.data.seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _make_generator() -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(cfg.data.seed)
        return g

    train_loader = DataLoader(
        TrajectoryDataset(train_split),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if cfg.train.num_workers > 0 else None,
        generator=_make_generator(),
        drop_last=False,
    )
    val_loader = DataLoader(
        TrajectoryDataset(val_split),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TrajectoryDataset(test_split),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_dataset(data: TrajectorySplit, path: Path) -> None:
    """
    Serialise a :class:`TrajectorySplit` to disk using :func:`torch.save`.

    Parameters
    ----------
    data:
        Split to save.
    path:
        Destination file path (typically ``*.pt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": data.states, "inputs": data.inputs, "contexts": data.contexts}, path)


def load_dataset(path: Path) -> TrajectorySplit:
    """
    Load a previously saved :class:`TrajectorySplit` from disk.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    TrajectorySplit
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return TrajectorySplit(
        states=payload["states"],
        inputs=payload["inputs"],
        contexts=payload["contexts"],
    )

