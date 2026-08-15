"""
src/nexora/adapter.py
=====================
NexoraCRNAdapter — integrates Nexora product features with the REAL CRN.

Real CRN API (from src/crn/model.py):
--------------------------------------
Constructor:
    ConvexReasoningNetwork(cfg: CRNConfig)

forward():
    def forward(self, x0: Tensor, inputs: Tensor) -> Tuple[Tensor, List[Tensor]]
    - x0     : (batch, state_dim)       — initial hidden state
    - inputs : (batch, T, input_dim)    — input sequence of length T
    - returns: (states, energies)
        states   : (batch, T+1, state_dim)  — full trajectory incl. x0
        energies : List[Tensor(batch,)]     — T per-step Lyapunov energies

Checkpoint format (.pt via torch.save):
    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,   (may be absent)
        "epoch":      int,
        "val_loss":   float,
        "config":     dict             (from CRNConfig.to_dict())
    }

Checkpoint load (method on model instance):
    model.load_checkpoint(path, optimizer=None, scheduler=None, device=None)
    returns: dict  {"epoch", "val_loss", "config", ...}

Adapter design
--------------
Product features (32-dim from features.py)
    ↓
Reshape to (batch=1, T=1, input_dim=32) — single-step sequence
    ↓
x0 = zeros(1, state_dim)
    ↓
ConvexReasoningNetwork.forward(x0, inputs)
    ↓
states: (1, 2, state_dim)  — [x0, x1]
    ↓
final_state = states[0, -1, :]  — x1: last state after 1 step
    ↓
Return structured Nexora result dict
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# ── Path setup: allow imports from src/ regardless of working directory ──
_SRC = Path(__file__).resolve().parent.parent   # .../src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import torch
    from torch import Tensor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from nexora.features import encode_product, NEXORA_INPUT_DIM

# Real CRN imports — never modified, used as-is
try:
    from crn.model import ConvexReasoningNetwork
    from crn.config import CRNConfig, ModelConfig
    _CRN_AVAILABLE = True
except ImportError as _crn_err:
    _CRN_AVAILABLE = False
    _CRN_IMPORT_ERROR = str(_crn_err)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent.parent
    / "checkpoints" / "nexora_crn_best.pt"
)

# Input dimension Nexora encodes into — must match CRN's input_dim
NEXORA_INPUT_DIM_EXPECTED = 32


class NexoraCRNAdapter:
    """
    Wraps the REAL ConvexReasoningNetwork for Nexora product inference.

    The adapter is intentionally thin:
    - It does NOT modify the CRN architecture.
    - It does NOT add any learned parameters.
    - It only handles tensor reshaping and config construction.

    Parameters
    ----------
    checkpoint_path : path to a .pt checkpoint saved by the real CRN training.
                      If None, uses DEFAULT_CHECKPOINT.
    device          : torch device string ("cpu", "cuda", etc.)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.device_str      = device
        self.is_trained      = False
        self.checkpoint_info: dict = {}
        self._load_error: Optional[str] = None

        if not _TORCH_AVAILABLE:
            self._load_error = "torch is not installed — CRN cannot run."
            self.model = None
            return

        if not _CRN_AVAILABLE:
            self._load_error = f"CRN import failed: {_CRN_IMPORT_ERROR}"
            self.model = None
            return

        self.device = torch.device(device)
        self.model, self.cfg = self._load_model()

    # ------------------------------------------------------------------
    # A. Constructor used
    # ------------------------------------------------------------------
    def _build_config_from_checkpoint(self, ckpt: dict) -> "CRNConfig":
        """
        Reconstruct CRNConfig from checkpoint dict.

        Real checkpoint stores config as a dict under key "config"
        (written by CheckpointManager.save → CRNConfig.to_dict()).
        """
        cfg = CRNConfig()                         # start with defaults
        raw = ckpt.get("config", {})

        # Restore model-level fields if present
        if "model" in raw and isinstance(raw["model"], dict):
            m = raw["model"]
            cfg.model.state_dim         = m.get("state_dim",         cfg.model.state_dim)
            cfg.model.input_dim         = m.get("input_dim",         cfg.model.input_dim)
            cfg.model.n_context_vectors = m.get("n_context_vectors", cfg.model.n_context_vectors)
            cfg.model.metric_type       = m.get("metric_type",       cfg.model.metric_type)
            cfg.model.solver            = m.get("solver",            cfg.model.solver)
            cfg.model.contraction_factor = m.get("contraction_factor", cfg.model.contraction_factor)
        elif raw:
            # Flat dict fallback (some checkpoint versions may flatten)
            cfg.model.state_dim         = raw.get("state_dim",         cfg.model.state_dim)
            cfg.model.input_dim         = raw.get("input_dim",         cfg.model.input_dim)
            cfg.model.n_context_vectors = raw.get("n_context_vectors", cfg.model.n_context_vectors)

        return cfg

    def _build_default_nexora_config(self) -> "CRNConfig":
        """
        Build a CRNConfig suitable for Nexora inference (no checkpoint).
        input_dim must equal NEXORA_INPUT_DIM = 32.
        """
        cfg = CRNConfig()
        cfg.model.input_dim = NEXORA_INPUT_DIM_EXPECTED
        # state_dim, n_context_vectors, etc. keep their CRNConfig defaults
        return cfg

    def _load_model(self) -> tuple["ConvexReasoningNetwork", "CRNConfig"]:
        """
        Load CRN from checkpoint or initialise with default config.

        B. Exact tensor shape: (batch=1, T=1, input_dim)
        D. Checkpoint load: model.load_checkpoint(path, device=self.device)
        """
        if self.checkpoint_path.exists():
            try:
                # Load raw dict to read config before constructing model
                raw_ckpt = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
                cfg = self._build_config_from_checkpoint(raw_ckpt)

                # Warn if input_dim mismatch
                if cfg.model.input_dim != NEXORA_INPUT_DIM_EXPECTED:
                    print(
                        f"[NexoraCRN] WARNING: checkpoint input_dim="
                        f"{cfg.model.input_dim} but Nexora encodes "
                        f"{NEXORA_INPUT_DIM_EXPECTED} dims. "
                        f"A linear projection will be applied."
                    )

                # D. Real constructor: ConvexReasoningNetwork(cfg)
                model = ConvexReasoningNetwork(cfg).to(self.device)

                # D. Real load method: model.load_checkpoint(path, device=...)
                ckpt_meta = model.load_checkpoint(
                    self.checkpoint_path,
                    device=self.device,
                )
                model.eval()

                self.is_trained = True
                self.checkpoint_info = {
                    "epoch":        ckpt_meta.get("epoch"),
                    "val_loss":     ckpt_meta.get("val_loss"),
                    "state_dim":    cfg.model.state_dim,
                    "input_dim":    cfg.model.input_dim,
                    "metric_type":  cfg.model.metric_type,
                    "dataset_type": "synthetic",   # honest label
                    "checkpoint":   str(self.checkpoint_path),
                }
                print(
                    f"[NexoraCRN] Checkpoint loaded: "
                    f"epoch={ckpt_meta.get('epoch')} "
                    f"val_loss={ckpt_meta.get('val_loss', float('inf')):.6f}"
                )
                return model, cfg

            except Exception as exc:
                self._load_error = (
                    f"Checkpoint load failed ({self.checkpoint_path}): {exc}"
                )
                print(f"[NexoraCRN] {self._load_error}")
                print("[NexoraCRN] Falling back to random initialisation.")

        # No checkpoint — use default config, randomly initialised
        cfg = self._build_default_nexora_config()
        model = ConvexReasoningNetwork(cfg).to(self.device)
        model.eval()
        print(
            f"[NexoraCRN] No checkpoint — random weights "
            f"(input_dim={cfg.model.input_dim}, state_dim={cfg.model.state_dim})"
        )
        return model, cfg

    # ------------------------------------------------------------------
    # B + C. Tensor shaping and forward call
    # ------------------------------------------------------------------
    def _make_input_tensors(
        self, feat_np
    ) -> tuple["Tensor", "Tensor"]:
        """
        Convert a 1-D numpy feature vector (NEXORA_INPUT_DIM,) into
        the shapes the real CRN forward() requires.

        Real CRN forward():
            x0     : (batch, state_dim)
            inputs : (batch, T, input_dim)

        We use batch=1, T=1, treating the product as a single-step sequence.
        If checkpoint input_dim ≠ NEXORA_INPUT_DIM, project with a fixed
        (non-learned) random projection for shape compatibility.

        Returns
        -------
        x0     : (1, state_dim)
        inputs : (1, 1, input_dim)   where input_dim = cfg.model.input_dim
        """
        import torch

        feat = torch.tensor(feat_np, dtype=torch.float32, device=self.device)
        # feat: (NEXORA_INPUT_DIM,) = (32,)

        crn_input_dim = self.cfg.model.input_dim
        crn_state_dim = self.cfg.model.state_dim

        # Projection if dims differ (deterministic fixed matrix)
        if crn_input_dim != NEXORA_INPUT_DIM_EXPECTED:
            gen = torch.Generator()
            gen.manual_seed(42)
            proj = torch.randn(
                NEXORA_INPUT_DIM_EXPECTED, crn_input_dim,
                generator=gen, device=self.device
            ) / (NEXORA_INPUT_DIM_EXPECTED ** 0.5)
            feat = feat @ proj                         # (crn_input_dim,)

        # x0: zeros initial state — (1, state_dim)
        x0 = torch.zeros(1, crn_state_dim, device=self.device)

        # inputs: (1, T=1, input_dim)
        inputs = feat.unsqueeze(0).unsqueeze(0)        # (1, 1, input_dim)

        return x0, inputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_inference(
        self,
        title:       str,
        description: str   = "",
        price:       Optional[float] = None,
        category:    str   = "other",
        url:         str   = "",
    ) -> dict:
        """
        Run the real CRN on a product and return a structured result.

        C. Real forward() output: (states, energies)
           states  : (batch=1, T+1=2, state_dim)
           energies: [Tensor(1,)]   — 1 energy per step

        Returns
        -------
        dict with keys:
            crn_status        : "trained" | "untrained_weights" | "error"
            state_dimension   : int  — CRN state_dim
            trajectory_length : int  — number of states (T+1 = 2 for T=1)
            final_state_norm  : float — L2 norm of x_T
            final_state       : list[float] — x_T as a list
            checkpoint_info   : dict — metadata from checkpoint
        """
        if self.model is None:
            return {
                "crn_status":        "error",
                "error":             self._load_error,
                "state_dimension":   0,
                "trajectory_length": 0,
                "final_state_norm":  0.0,
                "final_state":       [],
                "checkpoint_info":   {},
            }

        import torch

        # Encode product → 32-dim numpy array
        feat_np = encode_product(title, description, price, category, url)

        # Build tensors matching real CRN input contract
        x0, inputs = self._make_input_tensors(feat_np)

        try:
            with torch.no_grad():
                # C. Real forward() call:
                # (states, energies) = model(x0, inputs)
                states, energies = self.model(x0, inputs)
                # states : (1, T+1, state_dim) = (1, 2, state_dim)
                # energies: [Tensor(1,)]

            final_state: "Tensor" = states[0, -1, :]   # (state_dim,)
            trajectory_len = states.shape[1]            # T+1

            # Energy at step 1 (honest — only 1 step)
            step_energy = (
                float(energies[0][0].item()) if energies else None
            )

            return {
                "crn_status":        "trained" if self.is_trained else "untrained_weights",
                "state_dimension":   int(final_state.shape[0]),
                "trajectory_length": int(trajectory_len),
                "final_state_norm":  float(final_state.norm().item()),
                "step_energy":       step_energy,
                "final_state":       final_state.tolist(),
                "checkpoint_info":   self.checkpoint_info if self.is_trained else {
                    "note": (
                        "CRN running with random initialisation. "
                        "Train with src/crn/main.py to obtain a checkpoint."
                    ),
                    "load_error": self._load_error,
                },
            }

        except Exception as exc:
            return {
                "crn_status":        "error",
                "error":             str(exc),
                "state_dimension":   0,
                "trajectory_length": 0,
                "final_state_norm":  0.0,
                "final_state":       [],
                "checkpoint_info":   self.checkpoint_info,
            }

    # ------------------------------------------------------------------
    # Compatibility report (useful for debugging)
    # ------------------------------------------------------------------
    def compatibility_report(self) -> dict:
        """
        E. Return a dict describing exact API contract and any issues.
        """
        cfg = getattr(self, "cfg", None)
        return {
            "A_constructor": "ConvexReasoningNetwork(cfg: CRNConfig)",
            "B_input_x0_shape": f"(batch=1, state_dim={cfg.model.state_dim if cfg else '?'})",
            "B_input_inputs_shape": (
                f"(batch=1, T=1, input_dim={cfg.model.input_dim if cfg else '?'})"
            ),
            "C_output": "(states: Tensor(1,T+1,state_dim), energies: List[Tensor(1,)])",
            "D_checkpoint_load": "model.load_checkpoint(path, device=device)  → dict",
            "D_checkpoint_format": ".pt via torch.save({'model_state_dict','epoch','val_loss','config'})",
            "E_incompatibilities": (
                []
                if (_TORCH_AVAILABLE and _CRN_AVAILABLE and not self._load_error)
                else [
                    e for e in [
                        "torch not available" if not _TORCH_AVAILABLE else None,
                        f"CRN import error: {_CRN_IMPORT_ERROR}" if not _CRN_AVAILABLE else None,
                        self._load_error,
                    ] if e
                ]
            ),
            "nexora_input_dim":     NEXORA_INPUT_DIM_EXPECTED,
            "crn_input_dim":        cfg.model.input_dim if cfg else "?",
            "projection_needed":    (
                cfg.model.input_dim != NEXORA_INPUT_DIM_EXPECTED if cfg else "unknown"
            ),
            "is_trained":           self.is_trained,
        }
