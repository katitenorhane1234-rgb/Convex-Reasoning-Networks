"""
src/nexora/adapter.py
=====================
Loads the Nexora-CRN checkpoint and runs inference.

The CRN here acts as a structured reasoning / feature-transformation layer:

    product_features (32-dim)
           ↓
    ConvexReasoningNetwork
           ↓
    crn_state (64-dim) — convex-constrained latent representation
           ↓
    MarketingGenerator
           ↓
    campaign JSON
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
from torch import Tensor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from crn.model import ConvexReasoningNetwork
from crn.train import load_checkpoint
from nexora.features import encode_product, NEXORA_INPUT_DIM

DEFAULT_CHECKPOINT = Path(__file__).parent.parent.parent / "checkpoints" / "nexora_crn_best.pt"


class NexoraCRNAdapter:
    """
    Wraps the CRN for Nexora inference.

    If no checkpoint is found the model runs with randomly-initialised
    weights (clearly reported in the response).
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cpu") -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.is_trained = False
        self.checkpoint_info: dict = {}
        self.model = self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self) -> ConvexReasoningNetwork:
        if Path(self.checkpoint_path).exists():
            try:
                model, ckpt = load_checkpoint(self.checkpoint_path, device=self.device)
                self.is_trained = True
                self.checkpoint_info = {
                    "epoch": ckpt.get("epoch"),
                    "val_loss": ckpt.get("val_loss"),
                    "version": ckpt.get("version", "unknown"),
                    "dataset_type": ckpt.get("config", {}).get("dataset_type", "synthetic"),
                }
                return model
            except Exception as exc:
                print(f"[NexoraCRNAdapter] Warning: failed to load checkpoint: {exc}")

        # Fallback: uninitialised model
        model = ConvexReasoningNetwork(
            input_dim=NEXORA_INPUT_DIM,
            state_dim=64,
            num_prototypes=8,
            num_steps=4,
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    def run_inference(
        self,
        title: str,
        description: str = "",
        price: float | None = None,
        category: str = "other",
        url: str = "",
    ) -> dict:
        """
        Run CRN inference on a single product.

        Returns
        -------
        dict with keys:
            crn_status          : "trained" | "untrained_weights"
            state_dimension     : int
            trajectory_length   : int
            final_state_norm    : float   (L2 norm of final hidden state)
            final_state         : list[float]   (the actual CRN output)
        """
        feat = encode_product(title, description, price, category, url)
        feat = feat.unsqueeze(0).to(self.device)   # (1, input_dim)

        with torch.no_grad():
            out = self.model(feat)

        final_state: Tensor = out["final_state"][0]         # (state_dim,)
        trajectory:  Tensor = out["trajectory"][0]          # (T+1, state_dim)

        return {
            "crn_status": "trained" if self.is_trained else "untrained_weights",
            "state_dimension": final_state.shape[0],
            "trajectory_length": trajectory.shape[0],
            "final_state_norm": float(final_state.norm().item()),
            "final_state": final_state.tolist(),
            "checkpoint_info": self.checkpoint_info if self.is_trained else {
                "note": "Model running with random initialisation. Train first for meaningful representations."
            },
        }
