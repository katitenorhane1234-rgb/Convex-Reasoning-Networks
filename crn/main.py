"""
tests/test_main.py
==================
Interface-level tests for :mod:`main` (CLI entry point).

Verifies that:
1. The argument parser is correctly constructed.
2. Each sub-command is registered.
3. ``apply_cli_overrides`` propagates every supported flag to the config.
4. ``_resolve_config`` loads a JSON config file when ``--config`` is given.
5. The ``--dry-run`` flag on ``train`` exits with code 0 without training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from config import CRNConfig
from main import apply_cli_overrides, build_argument_parser, _resolve_config


# ---------------------------------------------------------------------------
# Argument parser structure
# ---------------------------------------------------------------------------

class TestBuildArgumentParser:
    def test_parser_is_not_none(self) -> None:
        parser = build_argument_parser()
        assert parser is not None

    def test_train_subcommand_registered(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--dry-run"])
        assert args.command == "train"

    def test_eval_subcommand_registered(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["eval"])
        assert args.command == "eval"

    def test_ablate_subcommand_registered(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["ablate"])
        assert args.command == "ablate"

    def test_all_subcommand_registered(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["all"])
        assert args.command == "all"

    def test_no_subcommand_raises(self) -> None:
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    @pytest.mark.parametrize("cmd", ["train", "eval", "ablate", "all"])
    def test_shared_name_flag(self, cmd: str) -> None:
        parser = build_argument_parser()
        args = parser.parse_args([cmd, "--name", "my_exp"])
        assert args.name == "my_exp"

    @pytest.mark.parametrize("cmd", ["train", "eval", "ablate", "all"])
    def test_shared_device_flag(self, cmd: str) -> None:
        parser = build_argument_parser()
        args = parser.parse_args([cmd, "--device", "cpu"])
        assert args.device == "cpu"

    @pytest.mark.parametrize("cmd", ["train", "eval", "ablate", "all"])
    def test_shared_seed_flag(self, cmd: str) -> None:
        parser = build_argument_parser()
        args = parser.parse_args([cmd, "--seed", "99"])
        assert args.seed == 99

    def test_train_epochs_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--epochs", "50"])
        assert args.epochs == 50

    def test_train_lr_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--lr", "0.001"])
        assert abs(args.lr - 0.001) < 1e-9

    def test_train_solver_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--solver", "pgd"])
        assert args.solver == "pgd"

    def test_train_metric_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--metric", "euclidean"])
        assert args.metric == "euclidean"

    def test_train_dry_run_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["train", "--dry-run"])
        assert args.dry_run is True

    def test_ablate_n_seeds_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["ablate", "--n-seeds", "5"])
        assert args.n_seeds == 5

    def test_ablate_metric_only_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["ablate", "--metric-only"])
        assert args.metric_only is True

    def test_ablate_solver_only_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["ablate", "--solver-only"])
        assert args.solver_only is True

    def test_eval_no_plots_flag(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["eval", "--no-plots"])
        assert args.no_plots is True

    def test_invalid_solver_choice_raises(self) -> None:
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--solver", "invalid_solver"])

    def test_invalid_metric_choice_raises(self) -> None:
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--metric", "invalid_metric"])


# ---------------------------------------------------------------------------
# apply_cli_overrides
# ---------------------------------------------------------------------------

class TestApplyCliOverrides:
    def _parse(self, argv: list[str]):
        return build_argument_parser().parse_args(argv)

    def test_name_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--name", "custom_name"])
        apply_cli_overrides(cfg, args)
        assert cfg.experiment_name == "custom_name"

    def test_device_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--device", "cpu"])
        apply_cli_overrides(cfg, args)
        assert cfg.train.device == "cpu"

    def test_epochs_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--epochs", "42"])
        apply_cli_overrides(cfg, args)
        assert cfg.train.epochs == 42

    def test_lr_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--lr", "0.01"])
        apply_cli_overrides(cfg, args)
        assert abs(cfg.train.learning_rate - 0.01) < 1e-9

    def test_seed_override_sets_both_data_and_train(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--seed", "77"])
        apply_cli_overrides(cfg, args)
        assert cfg.train.seed == 77
        assert cfg.data.seed == 77

    def test_solver_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--solver", "pgd"])
        apply_cli_overrides(cfg, args)
        assert cfg.model.solver == "pgd"

    def test_metric_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["train", "--metric", "euclidean"])
        apply_cli_overrides(cfg, args)
        assert cfg.model.metric_type == "euclidean"

    def test_n_seeds_override(self) -> None:
        cfg = CRNConfig()
        args = self._parse(["ablate", "--n-seeds", "7"])
        apply_cli_overrides(cfg, args)
        assert cfg.ablation.n_seeds == 7

    def test_none_args_do_not_override_defaults(self) -> None:
        cfg = CRNConfig()
        original_lr = cfg.train.learning_rate
        args = self._parse(["train"])  # no --lr given
        apply_cli_overrides(cfg, args)
        assert cfg.train.learning_rate == original_lr


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------

class TestResolveConfig:
    def test_returns_default_config_when_no_config_arg(self) -> None:
        args = build_argument_parser().parse_args(["train"])
        cfg = _resolve_config(args)
        assert isinstance(cfg, CRNConfig)

    def test_loads_json_config_from_file(self, tmp_path: Path) -> None:
        original_cfg = CRNConfig(experiment_name="loaded_from_file")
        config_path = tmp_path / "test_config.json"
        original_cfg.save(path=config_path)

        args = build_argument_parser().parse_args(["train", "--config", str(config_path)])
        cfg = _resolve_config(args)
        assert cfg.experiment_name == "loaded_from_file"

    def test_cli_overrides_loaded_config(self, tmp_path: Path) -> None:
        original_cfg = CRNConfig(experiment_name="original_name")
        config_path = tmp_path / "test_config.json"
        original_cfg.save(path=config_path)

        args = build_argument_parser().parse_args([
            "train", "--config", str(config_path), "--name", "override_name"
        ])
        cfg = _resolve_config(args)
        assert cfg.experiment_name == "override_name"


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------

class TestMainExitCodes:
    def test_dry_run_exits_zero(self, capsys) -> None:
        from main import main
        rc = main(["train", "--dry-run"])
        assert rc == 0

    def test_missing_checkpoint_for_eval_exits_nonzero(self, tmp_path: Path) -> None:
        from main import main
        rc = main([
            "eval",
            "--name", "nonexistent_experiment_xyz",
        ])
        assert rc != 0
