"""Cross-validation experiment runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from osteo_fracture.config import load_config
from train import run_training



def run_cross_validation(config_path: str, split_names: list[str]) -> None:
    """Run one training session per configured split name."""
    cfg = load_config(config_path)
    for split in split_names:
        cfg_fold = deepcopy(cfg)
        cfg_fold.data.split_name = split
        tmp_path = Path("configs") / f"_tmp_{split}.yaml"
        tmp_path.write_text(
            "\n".join(
                [
                    f"seed: {cfg_fold.seed}",
                    "data:",
                    f"  split_name: {cfg_fold.data.split_name}",
                ]
            )
        )
        run_training(str(tmp_path))
        tmp_path.unlink(missing_ok=True)
