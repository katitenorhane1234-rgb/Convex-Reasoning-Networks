
"""
utils.py
========
Shared utility functions for Convex Reasoning Networks.

This module collects small, single-responsibility helpers that are used
across multiple modules but do not belong to any single abstraction layer:

* **Reproducibility** — seeding, deterministic-mode activation
* **Device management** — automatic device selection and migration helpers
* **Parameter counting** — model size diagnostics
* **Logging** — structured experiment logger
* **Timing** — lightweight wall-clock profiler
* **Hardware info** — CPU / GPU metadata for reproducibility records
* **Tensor utilities** — safe operations, shape assertions, batch helpers
* **File I/O** — JSON serialisation, experiment directory management
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed all random number generators for reproducible execution.

    Sets seeds for:

    * Python built-in ``random``
    * NumPy (if installed)
    * PyTorch CPU and CUDA generators
    * ``PYTHONHASHSEED`` environment variable

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic:
        If True, set ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False``.  Slightly slower but
        fully reproducible on CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state() -> Dict[str, Any]:
    """
    Capture the current state of all random number generators.

    Returns
    -------
    dict
        Contains keys ``'python'``, ``'torch'``, ``'numpy'`` (if available),
        ``'cuda'`` (if CUDA is available).
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: Dict[str, Any]) -> None:
    """
    Restore the random number generator state captured by :func:`get_random_state`.

    Parameters
    ----------
    state:
        State dict produced by :func:`get_random_state`.
    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Priority (when ``device_str`` is None or ``'auto'``):

    1. CUDA (if available)
    2. MPS (Apple Silicon, if available)
    3. CPU

    Parameters
    ----------
    device_str:
        Device string such as ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ``'mps'``,
        or ``'auto'``.  Pass ``None`` for automatic selection.

    Returns
    -------
    torch.device
    """
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def move_batch_to_device(
    batch: Tuple[Tensor, ...],
    device: torch.device,
) -> Tuple[Tensor, ...]:
    """
    Move all tensors in a batch tuple to ``device``.

    Parameters
    ----------
    batch:
        Tuple of tensors (as returned by a DataLoader).
    device:
        Target device.

    Returns
    -------
    tuple
        Same structure as input, with all tensors on ``device``.
    """
    return tuple(t.to(device) for t in batch)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of (trainable) parameters in a model.

    Parameters
    ----------
    model:
        PyTorch module.
    trainable_only:
        If True (default), count only parameters with ``requires_grad=True``.

    Returns
    -------
    int
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Return a per-named-submodule parameter count.

    Parameters
    ----------
    model:
        PyTorch module.

    Returns
    -------
    dict
        Mapping from submodule name to parameter count (trainable only).
    """
    breakdown: Dict[str, int] = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        count = sum(
            p.numel()
            for p in module.parameters(recurse=False)
            if p.requires_grad
        )
        if count > 0:
            breakdown[name] = count
    return breakdown


def model_size_mb(model: nn.Module) -> float:
    """
    Estimate the memory footprint of a model's parameters in megabytes.

    Parameters
    ----------
    model:
        PyTorch module.

    Returns
    -------
    float
        Approximate size in MB.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently configured logger.

    The logger writes to stdout with a format that includes the timestamp,
    level, module name, and message.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    level:
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class ExperimentLogger:
    """
    Structured key-value logger for experiment metrics.

    Records metric snapshots to an in-memory list and flushes them to a
    JSONL file on each :meth:`log` call so that partial runs are preserved.

    Parameters
    ----------
    log_path:
        Path to the JSONL output file.
    experiment_name:
        Identifier included in every log record.
    """

    def __init__(self, log_path: Path, experiment_name: str) -> None:
        self.log_path = log_path
        self.experiment_name = experiment_name
        self._records: list[dict] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        """
        Append a metric snapshot to the log.

        Parameters
        ----------
        step:
            Global step counter (e.g. epoch number).
        metrics:
            Dict of metric name → value.
        **kwargs:
            Additional key-value pairs to include in the record.
        """
        record: Dict[str, Any] = {
            "experiment": self.experiment_name,
            "step": step,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": dict(metrics),
        }
        record.update(kwargs)
        self._records.append(record)
        self.flush()

    def flush(self) -> None:
        """Write all buffered records to the JSONL file."""
        with open(self.log_path, "w") as fh:
            for record in self._records:
                fh.write(json.dumps(record, default=str) + "\n")

    def to_dict_list(self) -> list[dict]:
        """Return all logged records as a list of dicts."""
        return list(self._records)

    @property
    def n_records(self) -> int:
        """Number of logged records."""
        return len(self._records)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """
    Context manager that measures and prints wall-clock time.

    Example
    -------
    >>> with timer("forward pass"):
    ...     y = model(x)
    forward pass: 12.3 ms

    Parameters
    ----------
    label:
        Human-readable label for the timed block.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        prefix = f"{label}: " if label else ""
        print(f"{prefix}{elapsed_ms:.1f} ms")


class Stopwatch:
    """
    Accumulating wall-clock stopwatch.

    Supports multiple laps; :meth:`elapsed` returns the total accumulated
    time across all laps.

    Example
    -------
    >>> sw = Stopwatch()
    >>> sw.start()
    >>> # ... work ...
    >>> sw.stop()
    >>> sw.elapsed_ms
    42.1
    """

    def __init__(self) -> None:
        self._total: float = 0.0
        self._t0: Optional[float] = None

    def start(self) -> None:
        """Start or resume the stopwatch."""
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the stopwatch and return the elapsed time for this lap (seconds).
        """
        if self._t0 is None:
            return 0.0
        lap = time.perf_counter() - self._t0
        self._total += lap
        self._t0 = None
        return lap

    def reset(self) -> None:
        """Reset accumulated time to zero."""
        self._total = 0.0
        self._t0 = None

    @property
    def elapsed_s(self) -> float:
        """Total accumulated time in seconds."""
        if self._t0 is not None:
            return self._total + (time.perf_counter() - self._t0)
        return self._total

    @property
    def elapsed_ms(self) -> float:
        """Total accumulated time in milliseconds."""
        return self.elapsed_s * 1000.0


# ---------------------------------------------------------------------------
# Hardware information
# ---------------------------------------------------------------------------


@dataclass
class HardwareInfo:
    """Snapshot of the execution hardware for reproducibility records."""

    hostname: str
    os_info: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    n_gpus: int
    gpu_names: list[str]
    cpu_count: int
    total_ram_gb: float


def get_hardware_info() -> HardwareInfo:
    """
    Collect hardware and software version information.

    Returns
    -------
    HardwareInfo
    """
    import socket

    cuda_available = torch.cuda.is_available()
    cuda_version: Optional[str] = torch.version.cuda if cuda_available else None
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [
        torch.cuda.get_device_name(i) for i in range(n_gpus)
    ]

    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        total_ram_gb = 0.0

    cpu_count = os.cpu_count() or 0

    return HardwareInfo(
        hostname=socket.gethostname(),
        os_info=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        n_gpus=n_gpus,
        gpu_names=gpu_names,
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
    )


def hardware_info_to_dict(info: HardwareInfo) -> Dict[str, Any]:
    """
    Serialise :class:`HardwareInfo` to a plain JSON-compatible dict.

    Parameters
    ----------
    info:
        Hardware information snapshot.

    Returns
    -------
    dict
    """
    return asdict(info)


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def assert_shape(tensor: Tensor, expected: Sequence[Optional[int]], name: str = "tensor") -> None:
    """
    Assert that a tensor has the expected shape.

    Pass ``None`` for a dimension that can be any size.

    Parameters
    ----------
    tensor:
        The tensor to check.
    expected:
        Expected shape, with ``None`` for wildcard dimensions.
    name:
        Variable name used in the error message.

    Raises
    ------
    AssertionError
        If the shapes do not match.
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)}D tensor, got {len(actual)}D "
            f"(shape {actual})"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is not None and a != e:
            raise AssertionError(
                f"{name}: dimension {i} expected {e}, got {a} "
                f"(full shape {actual}, expected {tuple(expected)})"
            )


def safe_cholesky(M: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Compute the Cholesky factorisation of M with automatic regularisation.

    If the plain Cholesky fails (M is numerically singular), adds
    ``eps * I`` and retries up to 5 times with exponentially increasing ε.

    Parameters
    ----------
    M:
        Symmetric positive semi-definite matrix of shape ``(d, d)``.
    eps:
        Initial regularisation increment.

    Returns
    -------
    Tensor
        Lower-triangular Cholesky factor L such that M ≈ L L^T.

    Raises
    ------
    torch.linalg.LinAlgError
        If the factorisation fails even after regularisation.
    """
    d = M.shape[0]
    for attempt in range(6):
        try:
            M_reg = M + (eps * (10 ** attempt)) * torch.eye(d, dtype=M.dtype, device=M.device)
            return torch.linalg.cholesky(M_reg)
        except torch.linalg.LinAlgError:
            if attempt == 5:
                raise
    # Unreachable, but satisfies type checker
    raise torch.linalg.LinAlgError("safe_cholesky failed after all retries")


def batch_outer(u: Tensor, v: Tensor) -> Tensor:
    """
    Compute batched outer products u ⊗ v.

    Parameters
    ----------
    u:
        Tensor of shape ``(batch, m)``.
    v:
        Tensor of shape ``(batch, n)``.

    Returns
    -------
    Tensor
        Outer product tensor of shape ``(batch, m, n)``.
    """
    return torch.bmm(u.unsqueeze(2), v.unsqueeze(1))


def spectral_norm(W: Tensor, n_iter: int = 10) -> Tensor:
    """
    Estimate the spectral norm (largest singular value) of W via power iteration.

    Parameters
    ----------
    W:
        Matrix of shape ``(m, n)``.
    n_iter:
        Number of power iterations (default 10).

    Returns
    -------
    Tensor
        Scalar spectral norm estimate.
    """
    m, n = W.shape
    # Initialise with a random unit vector
    v = torch.randn(n, 1, dtype=W.dtype, device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_iter):
        u = W @ v
        u = u / (u.norm() + 1e-12)
        v = W.t() @ u
        v = v / (v.norm() + 1e-12)
    sigma = (u.t() @ W @ v).squeeze()
    return sigma.abs()


# ---------------------------------------------------------------------------
# File I/O utilities
# ---------------------------------------------------------------------------


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """
    Save any JSON-serialisable object to a file.

    Parameters
    ----------
    obj:
        Object to serialise.
    path:
        Destination file path.
    indent:
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=str))


def load_json(path: Path) -> Any:
    """
    Load a JSON file and return the parsed object.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    Any
    """
    return json.loads(Path(path).read_text())


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        The same path, for convenient chaining.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_dir(cfg_or_name: Union["CRNConfig", str]) -> Path:  # noqa: F821
    """
    Return the results directory for an experiment, creating it if needed.

    Parameters
    ----------
    cfg_or_name:
        Either a :class:`~config.CRNConfig` instance or an experiment name string.

    Returns
    -------
    Path
    """
    from config import RESULTS_DIR

    if isinstance(cfg_or_name, str):
        name = cfg_or_name
    else:
        name = cfg_or_name.experiment_name
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
