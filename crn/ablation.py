"""
ablation.py
===========
Ablation study framework for Convex Reasoning Networks.

Runs a structured grid of experiments that isolate the contribution of each
architectural component:

**Metric ablation** — compares the learnable SPD metric against the fixed
Euclidean (identity) metric, holding everything else constant.

**Solver ablation** — compares the three solver implementations (Analytic,
PGD, Frank-Wolfe) on identical model instances, measuring both solution
quality and computational cost.

Each ablation cell is repeated over multiple seeds to obtain mean ± std
estimates.  Results are automatically saved as:

* ``results/<experiment_name>/ablation_metric.json``
* ``results/<experiment_name>/ablation_solver.json``
* ``results/<experiment_name>/ablation_table_metric.txt``
* ``results/<experiment_name>/ablation_table_solver.txt``

The primary entry point is :func:`run_ablation_study`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from config import CRNConfig, RESULTS_DIR
from crn import build_crn
