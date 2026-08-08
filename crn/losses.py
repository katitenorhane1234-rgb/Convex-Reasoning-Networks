
"""
Differentiable loss functions for Convex Reasoning Networks.

The CRN state update is

    x_{t+1} = Prox_C^M((I + A) x_t + B g_t).

This module contains the reconstruction objective and the regularizers that
support that update: simplex-weight validity, SPD-metric stability,
contractivity of the complete state transition, convex-hull projection
consistency, and optional L2 parameter regularization.  The functions are
independent of the training loop and preserve autograd for CPU and CUDA
execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


Scalar = Union[float, int]


@dataclass
class LossComponents:
    """
    Individual CRN loss terms and their weighted total.

    Every attribute is a scalar tensor.  The tensors are intentionally not
    detached so that ``total.backward()`` remains valid.  Use
    :meth:`as_dict` with ``detach=True`` for logging.
    """

    total: Tensor
    prediction: Tensor
    convexity: Tensor
    metric: Tensor
    contractivity: Tensor
    projection: Tensor
    l2: Tensor

    def as_dict(self, detach: bool = True) -> Dict[str, Tensor]:
        """
        Return all components in a dictionary.

        Parameters
        ----------
        detach:
            Detach values from autograd when true.  This is appropriate for
            metrics logging and serialization.
        """
        values = {
            "total": self.total,
            "prediction": self.prediction,
            "convexity": self.convexity,
            "metric": self.metric,
            "contractivity": self.contractivity,
            "projection": self.projection,
            "l2": self.l2,
        }
        return {key: value.detach() for key, value in values.items()} if detach else values

    def item_dict(self) -> Dict[str, float]:
        """Return detached scalar components as ordinary Python floats."""
        return {
            key: float(value.detach().cpu().item())
            for key, value in self.as_dict(detach=True).items()
        }


@dataclass(frozen=True)
class LossWeights:
    """
    Weights for the terms in :class:`CRNLoss`.

    The prediction term is dominant by default; regularizers are deliberately
    small so they stabilize training without replacing the data objective.
    """

    prediction: float = 1.0
    convexity: float = 1e-3
    metric: float = 1e-3
    contractivity: float = 1e-3
    projection: float = 1e-3
    l2: float = 1e-6

    def __post_init__(self) -> None:
        """Reject negative or non-finite loss weights."""
        for name, value in self.__dict__.items():
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} loss weight must be finite.")
            if value < 0:
                raise ValueError(f"{name} loss weight must be non-negative.")


def _as_scalar_tensor(value: Scalar, reference: Tensor) -> Tensor:
    """Create a scalar tensor matching a reference tensor's dtype and device."""
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _zero(reference: Optional[Tensor] = None) -> Tensor:
    """
    Return a differentiable scalar zero when a reference tensor is available.
    """
    if reference is not None:
        return reference.sum() * 0.0
    return torch.zeros((), dtype=torch.get_default_dtype())


def _validate_reduction(reduction: str) -> None:
    """Validate a supported reduction name."""
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
        )


def _trajectory_error(
    predicted: Tensor,
    target: Tensor,
    *,
    exclude_initial: bool,
) -> Tensor:
    """
    Return per-sample, per-time mean squared state error.

    Two-dimensional inputs are interpreted as one-step batches and produce a
    vector of shape ``(batch,)``. Three-dimensional inputs are interpreted as
    trajectories and produce ``(batch, time)``.
    """
    if not isinstance(predicted, Tensor) or not isinstance(target, Tensor):
        raise TypeError("predicted and target must be torch.Tensor instances.")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have identical shapes; "
            f"received {tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim not in (2, 3):
        raise ValueError(
            "predicted and target must have shape (batch, state_dim) or "
            "(batch, time, state_dim)."
        )
    if predicted.shape[-1] == 0:
        raise ValueError("The state dimension must be non-zero.")

    if predicted.ndim == 3 and exclude_initial:
        if predicted.shape[1] < 2:
            raise ValueError(
                "exclude_initial=True requires a trajectory with at least "
                "two time indices."
            )
        predicted = predicted[:, 1:, :]
        target = target[:, 1:, :]

    return (predicted - target).pow(2).mean(dim=-1)


def mse_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    exclude_initial: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute CRN prediction/reconstruction mean squared error.

    Parameters
    ----------
    predicted:
        A next-state batch of shape ``(batch, state_dim)`` or a trajectory of
        shape ``(batch, time, state_dim)``.
    target:
        Tensor with the same shape as ``predicted``.
    exclude_initial:
        For trajectory inputs, omit index zero because the initial state is
        supplied to the model rather than predicted.
    reduction:
        ``"mean"`` returns a scalar mean, ``"sum"`` returns a scalar sum, and
        ``"none"`` returns per-sample or per-sample/per-time errors.
    """
    _validate_reduction(reduction)
    error = _trajectory_error(
        predicted,
        target,
        exclude_initial=exclude_initial,
    )
    if reduction == "none":
        return error
    return error.mean() if reduction == "mean" else error.sum()


def prediction_mse_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    exclude_initial: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """Alias for :func:`mse_loss` with an explicit prediction-oriented name."""
    return mse_loss(
        predicted,
        target,
        exclude_initial=exclude_initial,
        reduction=reduction,
    )


def multi_step_rollout_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    exclude_initial: bool = True,
    discount: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute a horizon-weighted multi-step rollout loss.

    ``discount=1`` weights all predicted horizons equally.  A value below one
    emphasizes early predictions, while a value above one emphasizes later
    predictions.  The mean reduction normalizes the horizon weights to sum to
    one and then averages over the batch.
    """
    _validate_reduction(reduction)
    if discount <= 0 or not torch.isfinite(torch.tensor(float(discount))):
        raise ValueError("discount must be a finite positive number.")

    error = _trajectory_error(
        predicted,
        target,
        exclude_initial=exclude_initial,
    )
    if error.ndim == 1:
        return error if reduction == "none" else (
            error.mean() if reduction == "mean" else error.sum()
        )

    horizon = error.shape[1]
    weights = torch.pow(
        _as_scalar_tensor(discount, error),
        torch.arange(horizon, dtype=error.dtype, device=error.device),
    )
    weighted = error * weights.unsqueeze(0)
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum(dim=1).div(weights.sum().clamp_min(torch.finfo(error.dtype).eps)).mean()


def rollout_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    exclude_initial: bool = True,
    discount: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Alias for :func:`multi_step_rollout_loss`."""
    return multi_step_rollout_loss(
        predicted,
        target,
        exclude_initial=exclude_initial,
        discount=discount,
        reduction=reduction,
    )


def convexity_penalty(
    weights: Optional[Tensor],
    *,
    nonneg_weight: float = 1.0,
    sum_weight: float = 1.0,
    target_sum: float = 1.0,
    reduction: str = "mean",
    reference: Optional[Tensor] = None,
) -> Tensor:
    """
    Penalize invalid convex-combination coefficients.

    For coefficients ``alpha``, the penalty is

        mean(ReLU(-alpha)^2) + mean((sum(alpha) - target_sum)^2).

    The function accepts any leading dimensions and treats the final
    dimension as the coefficient dimension.  Passing ``None`` returns a
    differentiable zero when ``reference`` is supplied, otherwise a scalar
    zero.
    """
    _validate_reduction(reduction)
    if nonneg_weight < 0 or sum_weight < 0:
        raise ValueError("Convexity penalty weights must be non-negative.")
    if weights is None:
        return _zero(reference)
    if not isinstance(weights, Tensor) or weights.ndim < 1:
        raise ValueError("weights must have at least one dimension.")

    negative = F.relu(-weights).pow(2)
    sum_error = (weights.sum(dim=-1) - target_sum).pow(2)
    value = nonneg_weight * negative.mean(dim=-1) + sum_weight * sum_error
    if reduction == "none":
        return value
    return value.mean() if reduction == "mean" else value.sum()


def _locate_metric(model_or_metric: Any) -> Any:
    """Locate a metric object on either a metric or a CRN model."""
    if model_or_metric is None:
        return None
    if callable(getattr(model_or_metric, "matrix", None)):
        return model_or_metric
    for owner_name in ("_metric", "metric"):
        if hasattr(model_or_metric, owner_name):
            metric = getattr(model_or_metric, owner_name)
            if metric is not None:
                return metric
    for owner_name in ("_cell", "cell"):
        if hasattr(model_or_metric, owner_name):
            owner = getattr(model_or_metric, owner_name)
            if owner is not None and hasattr(owner, "metric"):
                return getattr(owner, "metric")
    return None


def _metric_matrix(metric: Any) -> Optional[Tensor]:
    """Obtain a metric matrix from common CRN metric implementations."""
    if metric is None:
        return None
    matrix_method = getattr(metric, "matrix", None)
    if callable(matrix_method):
        matrix = matrix_method()
    elif isinstance(metric, Tensor):
        matrix = metric
    else:
        matrix = None
    if matrix is None:
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The metric matrix must be square.")
    return (matrix + matrix.transpose(-1, -2)) * 0.5


def metric_regularization(
    model_or_metric: Any,
    *,
    metric_eps: Optional[float] = None,
    min_eigenvalue: Optional[float] = None,
    max_condition_number: Optional[float] = 100.0,
    target_log_determinant: Optional[float] = None,
    condition_weight: float = 1.0,
    logdet_weight: float = 1e-2,
) -> Tensor:
    """
    Regularize a learned SPD metric.

    The penalty contains a squared eigenvalue-floor hinge, an optional
    condition-number hinge, and an optional log-determinant target penalty.
    When ``min_eigenvalue`` is omitted, ``metric_eps`` is read from the metric
    or model configuration when available and otherwise defaults to ``1e-4``.
    """
    metric = _locate_metric(model_or_metric)
    matrix = _metric_matrix(metric)
    if matrix is None:
        return _zero(model_or_metric if isinstance(model_or_metric, Tensor) else None)
    if condition_weight < 0 or logdet_weight < 0:
        raise ValueError("Metric regularization weights must be non-negative.")

    if metric_eps is None:
        metric_eps = getattr(metric, "eps", None)
    if metric_eps is None:
        cfg = getattr(model_or_metric, "cfg", None)
        model_cfg = getattr(cfg, "model", cfg)
        metric_eps = getattr(model_cfg, "metric_eps", 1e-4)
    if min_eigenvalue is None:
        min_eigenvalue = float(metric_eps)
    if min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be non-negative.")
    if max_condition_number is not None and max_condition_number <= 0:
        raise ValueError("max_condition_number must be positive.")

    eigenvalues = torch.linalg.eigvalsh(matrix)
    floor = F.relu(
        _as_scalar_tensor(min_eigenvalue, eigenvalues) - eigenvalues
    ).pow(2).mean()
    result = floor

    if max_condition_number is not None:
        smallest = eigenvalues.min().clamp_min(torch.finfo(eigenvalues.dtype).tiny)
        condition = eigenvalues.max() / smallest
        result = result + condition_weight * F.relu(
            condition - _as_scalar_tensor(max_condition_number, condition)
        ).pow(2)

    if target_log_determinant is not None:
        sign, logdet = torch.linalg.slogdet(matrix)
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
        result = result + logdet_weight * (
            logdet - _as_scalar_tensor(target_log_determinant, logdet)
        ).pow(2)
    return result


def _locate_transition(model_or_transition: Any) -> Optional[Tensor]:
    """Locate a CRN residual transition tensor from a model or tensor."""
    if isinstance(model_or_transition, Tensor):
        return model_or_transition
    if model_or_transition is None:
        return None

    candidates = [model_or_transition]
    for owner_name in ("_cell", "cell"):
        owner = getattr(model_or_transition, owner_name, None)
        if owner is not None:
            candidates.insert(0, owner)

    for owner in candidates:
        if hasattr(owner, "A"):
            value = getattr(owner, "A")
            if isinstance(value, Tensor):
                return value
        for name in ("_A_raw", "A_raw"):
            value = getattr(owner, name, None)
            if isinstance(value, Tensor):
                return value
    return None


def contractivity_penalty(
    model_or_transition: Any,
    *,
    contraction_factor: Optional[float] = None,
    margin: float = 0.0,
    include_identity: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Penalize a CRN transition whose spectral norm exceeds its bound.

    By default the full map ``I + A`` is measured because that is the map used
    by the CRN recurrence.  Set ``include_identity=False`` when regularizing a
    standalone matrix that is already the complete transition map.
    """
    _validate_reduction(reduction)
    transition = _locate_transition(model_or_transition)
    if transition is None:
        return _zero()
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("The transition matrix must be square.")

    if contraction_factor is None:
        cfg = getattr(model_or_transition, "cfg", None)
        model_cfg = getattr(cfg, "model", cfg)
        contraction_factor = getattr(model_cfg, "contraction_factor", 0.9)
    if contraction_factor <= 0:
        raise ValueError("contraction_factor must be positive.")
    if margin < 0 or margin >= contraction_factor:
        raise ValueError("margin must satisfy 0 <= margin < contraction_factor.")

    matrix = transition
    if include_identity:
        matrix = matrix + torch.eye(
            matrix.shape[0],
            dtype=matrix.dtype,
            device=matrix.device,
        )
    norm = torch.linalg.matrix_norm(matrix, ord=2)
    excess = F.relu(norm - _as_scalar_tensor(contraction_factor - margin, norm))
    value = excess.pow(2)
    return value if reduction != "none" else value


def projection_consistency_loss(
    states: Optional[Tensor],
    *,
    convex_set: Optional[Any] = None,
    model: Optional[nn.Module] = None,
    contexts: Optional[Tensor] = None,
    context_weight: float = 0.0,
    include_initial: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """
    Penalize distance between states and their convex-hull projections.

    An optional ``contexts`` tensor of simplex coefficients can add a second
    penalty between states and the decoded context combinations.  The
    coefficient is zero by default because context weights may be auxiliary
    dataset annotations rather than causal state inputs.
    """
    _validate_reduction(reduction)
    if states is None:
        return _zero()
    if states.ndim != 3:
        raise ValueError("states must have shape (batch, time, state_dim).")
    if context_weight < 0:
        raise ValueError("context_weight must be non-negative.")

    if convex_set is None and model is not None:
        convex_set = getattr(model, "_convex_set", None)
        if convex_set is None:
            convex_set = getattr(model, "convex_set", None)
        if convex_set is None:
            cell = getattr(model, "_cell", getattr(model, "cell", None))
            convex_set = getattr(cell, "convex_set", None)
    if convex_set is None or not callable(getattr(convex_set, "project", None)):
        return _zero(states)

    selected = states if include_initial else states[:, 1:, :]
    if selected.shape[1] == 0:
        return _zero(states)
    flat = selected.reshape(-1, selected.shape[-1])
    projected = convex_set.project(flat)
    error = (flat - projected).pow(2).mean(dim=-1).reshape(
        selected.shape[0], selected.shape[1]
    )

    if contexts is not None and context_weight > 0:
        if contexts.ndim == 2:
            contexts = contexts.unsqueeze(1)
        if contexts.ndim != 3 or contexts.shape[0] != states.shape[0]:
            raise ValueError("contexts must have shape (batch, time, n_contexts).")
        if contexts.shape[1] == states.shape[1] and not include_initial:
            contexts = contexts[:, 1:, :]
        if contexts.shape[1] != selected.shape[1]:
            raise ValueError("contexts must align with the selected state times.")
        prototypes = getattr(convex_set, "prototypes", None)
        if prototypes is None:
            raise AttributeError("convex_set must expose learnable prototypes.")
        decoded = contexts @ prototypes
        error = error + context_weight * (selected - decoded).pow(2).mean(dim=-1)

    if reduction == "none":
        return error
    return error.mean() if reduction == "mean" else error.sum()


def convex_hull_projection_consistency_loss(
    states: Tensor,
    model: Optional[nn.Module] = None,
    *,
    convex_set: Optional[Any] = None,
    contexts: Optional[Tensor] = None,
    context_weight: float = 0.0,
    include_initial: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Public alias for :func:`projection_consistency_loss`."""
    return projection_consistency_loss(
        states,
        convex_set=convex_set,
        model=model,
        contexts=contexts,
        context_weight=context_weight,
        include_initial=include_initial,
        reduction=reduction,
    )


def l2_weight_regularization(
    model: Optional[nn.Module],
    *,
    include_biases: bool = True,
    normalize: bool = True,
) -> Tensor:
    """
    Compute differentiable L2 regularization over trainable parameters.

    Parameters
    ----------
    model:
        Model whose trainable parameters should be regularized.  ``None``
        returns a scalar zero.
    include_biases:
        Exclude parameters with ``"bias"`` in their name when false.
    normalize:
        Divide by the number of included scalar parameters when true.
    """
    if model is None:
        return _zero()
    terms = []
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not include_biases and "bias" in name.lower():
            continue
        terms.append(parameter.pow(2).sum())
        count += parameter.numel()
    if not terms:
        return _zero(next(model.parameters(), None))
    result = torch.stack(terms).sum()
    return result / count if normalize else result


def l2_regularization(
    model: Optional[nn.Module],
    *,
    include_biases: bool = True,
    normalize: bool = True,
) -> Tensor:
    """Alias for :func:`l2_weight_regularization`."""
    return l2_weight_regularization(
        model,
        include_biases=include_biases,
        normalize=normalize,
    )


def _config_value(cfg: Optional[Any], names: Sequence[str], default: Any) -> Any:
    """Read an optional value from cfg.loss, cfg, cfg.train, or cfg.model."""
    if cfg is None:
        return default
    containers = [
        getattr(cfg, "loss", None),
        cfg,
        getattr(cfg, "train", None),
        getattr(cfg, "model", None),
    ]
    for container in containers:
        if container is None:
            continue
        for name in names:
            if hasattr(container, name):
                value = getattr(container, name)
                if value is not None:
                    return value
    return default


class CRNLoss(nn.Module):
    """
    Configurable composite loss for CRN training.

    The forward result is a :class:`LossComponents` object, allowing a
    training loop to backpropagate through ``result.total`` while logging
    every unweighted component from ``result.item_dict()``.
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        model: Optional[nn.Module] = None,
        *,
        prediction_weight: Optional[float] = None,
        convexity_weight: Optional[float] = None,
        metric_weight: Optional[float] = None,
        contractivity_weight: Optional[float] = None,
        projection_weight: Optional[float] = None,
        l2_weight: Optional[float] = None,
        rollout_weight: float = 0.0,
        rollout_discount: float = 1.0,
        min_metric_eigenvalue: Optional[float] = None,
        max_metric_condition: Optional[float] = 100.0,
        context_weight: float = 0.0,
        include_initial_in_projection: bool = False,
        include_biases_in_l2: bool = True,
        normalize_l2: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model

        self.weights = LossWeights(
            prediction=float(_config_value(
                cfg,
                ("prediction_weight", "prediction_loss_weight"),
                1.0 if prediction_weight is None else prediction_weight,
            ) if prediction_weight is None else prediction_weight),
            convexity=float(_config_value(
                cfg,
                ("convexity_weight", "convexity_loss_weight"),
                1e-3 if convexity_weight is None else convexity_weight,
            ) if convexity_weight is None else convexity_weight),
            metric=float(_config_value(
                cfg,
                ("metric_weight", "metric_regularization_weight"),
                1e-3 if metric_weight is None else metric_weight,
            ) if metric_weight is None else metric_weight),
            contractivity=float(_config_value(
                cfg,
                ("contractivity_weight", "contraction_weight"),
                1e-3 if contractivity_weight is None else contractivity_weight,
            ) if contractivity_weight is None else contractivity_weight),
            projection=float(_config_value(
                cfg,
                ("projection_weight", "convex_hull_weight"),
                1e-3 if projection_weight is None else projection_weight,
            ) if projection_weight is None else projection_weight),
            l2=float(_config_value(
                cfg,
                ("l2_weight", "weight_regularization"),
                1e-6 if l2_weight is None else l2_weight,
            ) if l2_weight is None else l2_weight),
        )
        self.rollout_weight = float(rollout_weight)
        self.rollout_discount = float(rollout_discount)
        self.min_metric_eigenvalue = min_metric_eigenvalue
        self.max_metric_condition = max_metric_condition
        self.context_weight = context_weight
        self.include_initial_in_projection = include_initial_in_projection
        self.include_biases_in_l2 = include_biases_in_l2
        self.normalize_l2 = normalize_l2
        if self.rollout_weight < 0:
            raise ValueError("rollout_weight must be non-negative.")
        if self.rollout_discount <= 0:
            raise ValueError("rollout_discount must be positive.")

    def forward(
        self,
        predicted: Tensor,
        target: Tensor,
        *,
        model: Optional[nn.Module] = None,
        context_weights: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
    ) -> LossComponents:
        """
        Compute the complete weighted CRN objective.

        Parameters
        ----------
        predicted:
            Predicted next state or state trajectory.
        target:
            Target next state or trajectory.
        model:
            Optional model overriding the constructor model.
        context_weights, contexts:
            Optional simplex coefficients.  Both names are accepted.
        """
        active_model = model if model is not None else self.model
        prediction = mse_loss(predicted, target)
        rollout = multi_step_rollout_loss(
            predicted,
            target,
            discount=self.rollout_discount,
        ) if predicted.ndim == 3 and self.rollout_weight > 0 else _zero(predicted)

        weights = context_weights if context_weights is not None else contexts
        convexity = convexity_penalty(
            weights,
            reference=predicted,
        )
        metric = metric_regularization(active_model)
        contractivity = contractivity_penalty(active_model)
        projection = projection_consistency_loss(
            predicted if predicted.ndim == 3 else None,
            model=active_model,
            contexts=weights,
            context_weight=self.context_weight,
            include_initial=self.include_initial_in_projection,
        )
        l2 = l2_weight_regularization(
            active_model,
            include_biases=self.include_biases_in_l2,
            normalize=self.normalize_l2,
        )

        total = (
            self.weights.prediction * prediction
            + self.rollout_weight * rollout
            + self.weights.convexity * convexity
            + self.weights.metric * metric
            + self.weights.contractivity * contractivity
            + self.weights.projection * projection
            + self.weights.l2 * l2
        )
        return LossComponents(
            total=total,
            prediction=prediction,
            convexity=convexity,
            metric=metric,
            contractivity=contractivity,
            projection=projection,
            l2=l2,
        )

    def compute(
        self,
        predicted: Tensor,
        target: Tensor,
        *,
        model: Optional[nn.Module] = None,
        context_weights: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
    ) -> LossComponents:
        """Explicit alias for :meth:`forward`."""
        return self.forward(
            predicted,
            target,
            model=model,
            context_weights=context_weights,
            contexts=contexts,
        )

    def extra_repr(self) -> str:
        """Return the configured composite-loss weights."""
        return (
            f"prediction={self.weights.prediction}, "
            f"rollout={self.rollout_weight}, "
            f"convexity={self.weights.convexity}, "
            f"metric={self.weights.metric}, "
            f"contractivity={self.weights.contractivity}, "
            f"projection={self.weights.projection}, "
            f"l2={self.weights.l2}"
        )


def total_crn_loss(
    predicted: Tensor,
    target: Tensor,
    *,
    model: Optional[nn.Module] = None,
    cfg: Optional[Any] = None,
    context_weights: Optional[Tensor] = None,
    contexts: Optional[Tensor] = None,
    prediction_weight: Optional[float] = None,
    convexity_weight: Optional[float] = None,
    metric_weight: Optional[float] = None,
    contractivity_weight: Optional[float] = None,
    projection_weight: Optional[float] = None,
    l2_weight: Optional[float] = None,
    return_components: bool = True,
) -> Union[LossComponents, Tensor]:
    """
    Functional entry point for the complete CRN loss.

    By default this returns :class:`LossComponents`.  Set
    ``return_components=False`` when only the scalar total is needed.
    """
    criterion = CRNLoss(
        cfg=cfg,
        model=model,
        prediction_weight=prediction_weight,
        convexity_weight=convexity_weight,
        metric_weight=metric_weight,
        contractivity_weight=contractivity_weight,
        projection_weight=projection_weight,
        l2_weight=l2_weight,
    )
    result = criterion(
        predicted,
        target,
        context_weights=context_weights,
        contexts=contexts,
    )
    return result if return_components else result.total
