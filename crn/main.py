"""
main.py
=======
Command-line entry point for Convex Reasoning Networks.

Provides a single ``crn`` CLI with four sub-commands:

+------------+------------------------------------------------------------+
| Sub-command| Action                                                     |
+============+============================================================+
| ``train``  | Train a CRN from scratch (or resume from a checkpoint).    |
| ``eval``   | Evaluate a trained model and generate the report.          |
| ``ablate`` | Run the full ablation study.                               |
| ``all``    | Run train → eval → ablate → figures in one shot.           |
+------------+------------------------------------------------------------+

Usage examples
--------------
.. code-block:: bash

    # Train with default configuration
    python main.py train

    # Train with custom experiment name and device
    python main.py train --name my_exp --device cuda

    # Evaluate a trained model
    python main.py eval --name my_exp

    # Run the ablation study
    python main.py ablate --name my_exp

    # Run everything end-to-end
    python main.py all --name my_exp --device cuda

    # Print the current configuration (no training)
    python main.py train --dry-run

All results are written to::

    checkpoints/<experiment_name>/
    results/<experiment_name>/
    figures/<experiment_name>/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    CRNConfig,
    DataConfig,
    ModelConfig,
    SolverConfig,
    TrainConfig,
    EvalConfig,
    AblationConfig,
    RESULTS_DIR,
    FIGURES_DIR,
)
from ablation import run_ablation_study
from crn import build_crn
from dataset import build_dataloaders
from evaluate import evaluate
from plots import generate_all_figures
from train import train
from utils import get_hardware_info, hardware_info_to_dict, get_logger, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration overrides from CLI arguments
# ---------------------------------------------------------------------------


def apply_cli_overrides(cfg: CRNConfig, args: argparse.Namespace) -> CRNConfig:
    """
    Apply command-line argument overrides to a :class:`CRNConfig`.

    Only fields explicitly provided on the command line are overridden;
    all others retain their default values.

    Parameters
    ----------
    cfg:
        Base configuration to modify (in-place).
    args:
        Parsed namespace from :func:`build_argument_parser`.

    Returns
    -------
    CRNConfig
        The modified configuration (same object).
    """
    if hasattr(args, "name") and args.name is not None:
        cfg.experiment_name = args.name
    if hasattr(args, "device") and args.device is not None:
        cfg.train.device = args.device
    if hasattr(args, "epochs") and args.epochs is not None:
        cfg.train.epochs = args.epochs
    if hasattr(args, "batch_size") and args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if hasattr(args, "lr") and args.lr is not None:
        cfg.train.learning_rate = args.lr
    if hasattr(args, "seed") and args.seed is not None:
        cfg.train.seed = args.seed
        cfg.data.seed = args.seed
    if hasattr(args, "solver") and args.solver is not None:
        cfg.model.solver = args.solver
    if hasattr(args, "metric") and args.metric is not None:
        cfg.model.metric_type = args.metric
    if hasattr(args, "n_seeds") and args.n_seeds is not None:
        cfg.ablation.n_seeds = args.n_seeds
    return cfg


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the top-level argument parser and all sub-command parsers.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="crn",
        description="Convex Reasoning Networks — research implementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Shared parent parser ----
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--name", type=str, default=None, metavar="EXPERIMENT_NAME",
        help="Experiment name used in checkpoint and results paths.",
    )
    parent.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help="Torch device string: 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parent.add_argument(
        "--config", type=Path, default=None, metavar="PATH",
        help="Path to a JSON config file (overrides defaults).",
    )
    parent.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="Master random seed.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = subparsers.add_parser(
        "train",
        parents=[parent],
        help="Train a CRN from scratch (or resume from a checkpoint).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_p.add_argument("--epochs", type=int, default=None, metavar="N")
    train_p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    train_p.add_argument("--lr", type=float, default=None, metavar="LEARNING_RATE")
    train_p.add_argument(
        "--solver",
        choices=["analytic", "pgd", "frank_wolfe"],
        default=None,
        help="Proximal solver for training.",
    )
    train_p.add_argument(
        "--metric",
        choices=["spd", "euclidean"],
        default=None,
        help="Metric type for training.",
    )
    train_p.add_argument(
        "--resume", type=Path, default=None, metavar="CHECKPOINT",
        help="Resume training from this checkpoint file.",
    )
    train_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved configuration and exit without training.",
    )

    # ---- eval ----
    eval_p = subparsers.add_parser(
        "eval",
        parents=[parent],
        help="Evaluate a trained CRN and save the report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_p.add_argument(
        "--checkpoint", type=Path, default=None, metavar="PATH",
        help="Checkpoint to evaluate (defaults to best.pt for the experiment).",
    )
    eval_p.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation.",
    )

    # ---- ablate ----
    ablate_p = subparsers.add_parser(
        "ablate",
        parents=[parent],
        help="Run the ablation study (metric + solver).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ablate_p.add_argument(
        "--n-seeds", dest="n_seeds", type=int, default=None, metavar="N",
        help="Number of random seeds per ablation cell.",
    )
    ablate_p.add_argument(
        "--metric-only", action="store_true",
        help="Run only the metric ablation.",
    )
    ablate_p.add_argument(
        "--solver-only", action="store_true",
        help="Run only the solver ablation.",
    )

    # ---- all ----
    all_p = subparsers.add_parser(
        "all",
        parents=[parent],
        help="Run the full pipeline: train → eval → ablate → figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_p.add_argument("--epochs", type=int, default=None)
    all_p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    all_p.add_argument("--lr", type=float, default=None)
    all_p.add_argument("--n-seeds", dest="n_seeds", type=int, default=None)

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_train(cfg: CRNConfig, args: argparse.Namespace) -> int:
    """
    Execute the ``train`` sub-command.

    Parameters
    ----------
    cfg:
        Resolved experiment configuration.
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    if getattr(args, "dry_run", False):
        print("\n--- Resolved Configuration ---")
        print(cfg.to_json())
        return 0

    logger.info("Starting training: %s", cfg.experiment_name)
    logger.info("Configuration fingerprint: %s", cfg.fingerprint())

    # Persist configuration
    cfg_path = cfg.save()
    logger.info("Configuration saved to: %s", cfg_path)

    # Record hardware info
    hw = get_hardware_info()
    hw_path = RESULTS_DIR / cfg.experiment_name / "hardware_info.json"
    hw_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(hardware_info_to_dict(hw), hw_path)
    logger.info("Hardware: %s", hw.os_info)
    if hw.cuda_available:
        logger.info("GPU(s): %s", ", ".join(hw.gpu_names))

    checkpoint_path = getattr(args, "resume", None)
    if checkpoint_path is not None:
        from train import resume_training
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        model, result = resume_training(cfg, checkpoint_path)
    else:
        model, result = train(cfg)

    logger.info("Training complete.")
    logger.info(
        "Best validation loss: %.6f at epoch %d",
        result.best_val_loss,
        result.best_epoch,
    )
    if result.stopped_early:
        logger.info("Early stopping triggered.")
    logger.info("Total time: %.1f s", result.total_time_s)
    logger.info("Checkpoint: %s", result.checkpoint_path)
    return 0


def cmd_eval(cfg: CRNConfig, args: argparse.Namespace) -> int:
    """
    Execute the ``eval`` sub-command.

    Parameters
    ----------
    cfg:
        Resolved experiment configuration.
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_path("best")

    if not Path(checkpoint_path).exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    logger.info("Evaluating: %s", cfg.experiment_name)
    logger.info("Checkpoint: %s", checkpoint_path)

    report = evaluate(cfg, checkpoint_path=Path(checkpoint_path))
    report_path = report.save()
    logger.info("Evaluation report saved to: %s", report_path)

    print("\n" + report.summary_table())

    if not getattr(args, "no_plots", False):
        logger.info("Generating figures …")
        # Load training result for loss curves (if available)
        _maybe_generate_eval_figures(cfg, report)

    return 0


def cmd_ablate(cfg: CRNConfig, args: argparse.Namespace) -> int:
    """
    Execute the ``ablate`` sub-command.

    Parameters
    ----------
    cfg:
        Resolved experiment configuration.
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    metric_only = getattr(args, "metric_only", False)
    solver_only = getattr(args, "solver_only", False)

    if metric_only:
        cfg.ablation.run_metric_ablation = True
        cfg.ablation.run_solver_ablation = False
    elif solver_only:
        cfg.ablation.run_metric_ablation = False
        cfg.ablation.run_solver_ablation = True

    logger.info("Running ablation study: %s", cfg.experiment_name)
    metric_results, solver_results = run_ablation_study(cfg, verbose=True)
    logger.info("Ablation study complete.")
    return 0


def cmd_all(cfg: CRNConfig, args: argparse.Namespace) -> int:
    """
    Execute the ``all`` sub-command (full pipeline).

    Parameters
    ----------
    cfg:
        Resolved experiment configuration.
    args:
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    t_start = time.monotonic()

    # 1. Train
    logger.info("=== PHASE 1/3: Training ===")
    model, result = train(cfg)

    # 2. Evaluate
    logger.info("=== PHASE 2/3: Evaluation ===")
    report = evaluate(cfg, model=model)
    print("\n" + report.summary_table())

    # 3. Ablation study
    logger.info("=== PHASE 3/3: Ablation Study ===")
    metric_results, solver_results = run_ablation_study(cfg, verbose=True)

    # 4. Figures
    logger.info("=== Generating all figures ===")
    saved_paths = generate_all_figures(
        cfg,
        result=result,
        report=report,
        metric_ablation=metric_results or None,
        solver_ablation=solver_results or None,
        verbose=True,
    )
    logger.info("Saved %d figures.", len(saved_paths))

    total_time = time.monotonic() - t_start
    logger.info("Full pipeline complete in %.1f s.", total_time)
    return 0


# ---------------------------------------------------------------------------
# Helper: figures after standalone eval
# ---------------------------------------------------------------------------


def _maybe_generate_eval_figures(cfg: CRNConfig, report: object) -> None:
    """
    Attempt to generate evaluation figures.

    Skips silently if training result artefacts are not available.
    """
    try:
        import matplotlib  # confirm matplotlib is installed
        from plots import plot_eigenvalue_spectrum, plot_condition_number

        fig_dir = FIGURES_DIR / cfg.experiment_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_eigenvalue_spectrum(report, cfg, save=True)
        logger.info("Eigenvalue spectrum figure saved.")
    except Exception as exc:
        logger.warning("Could not generate eval figures: %s", exc)


# ---------------------------------------------------------------------------
# Configuration loading helper
# ---------------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> CRNConfig:
    """
    Load or construct a :class:`CRNConfig` and apply CLI overrides.

    Parameters
    ----------
    args:
        Parsed CLI namespace.

    Returns
    -------
    CRNConfig
    """
    config_path = getattr(args, "config", None)
    if config_path is not None and Path(config_path).exists():
        cfg = CRNConfig.load(Path(config_path))
        logger.info("Configuration loaded from: %s", config_path)
    else:
        cfg = CRNConfig()

    cfg = apply_cli_overrides(cfg, args)
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse arguments, resolve configuration, and dispatch to the appropriate
    sub-command handler.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when None).

    Returns
    -------
    int
        Exit code.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    cfg = _resolve_config(args)

    dispatch = {
        "train": cmd_train,
        "eval": cmd_eval,
        "ablate": cmd_ablate,
        "all": cmd_all,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
