"""Simple hyperparameter search orchestration."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

from train import run_training



def run_hyperparameter_search(config_path: str, learning_rates: list[float], model_names: list[str]) -> None:
    """Run grid search over learning rates and model names."""
    base = yaml.safe_load(Path(config_path).read_text())
    for lr, model_name in product(learning_rates, model_names):
        cfg = dict(base)
        cfg.setdefault("model", {})["learning_rate"] = lr
        cfg.setdefault("model", {})["name"] = model_name
        cfg.setdefault("logging", {})["run_name"] = f"{model_name}-lr{lr}"
        tmp_path = Path("configs") / f"_tmp_{model_name}_{lr}.yaml"
        tmp_path.write_text(yaml.safe_dump(cfg))
        run_training(str(tmp_path))
        tmp_path.unlink(missing_ok=True)
