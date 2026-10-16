import copy
from pathlib import Path
from typing import Any
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively deep merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file, resolving 'extends' recursively."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "extends" in cfg:
        parent_name = cfg.pop("extends")
        parent_path = path.parent / parent_name
        if not parent_path.exists():
            parent_path = Path(parent_name)
        parent_cfg = load_config(parent_path)
        cfg = deep_merge(parent_cfg, cfg)

    return cfg


def get(cfg: dict, path: str, default: Any = None) -> Any:
    """Retrieve a nested value from cfg using dot-notation, e.g. 'a.b.c'."""
    keys = path.split(".")
    curr = cfg
    for key in keys:
        if not isinstance(curr, dict) or key not in curr:
            return default
        curr = curr[key]
    return curr


def validate_config(cfg: dict) -> None:
    """
    Validate config constraints:
    - dt <= 0
    - any speed/accel <= 0
    - decoy_fraction outside [0, 1]
    - max_salvo_owners_k not in {1, 2}
    - num_parallel_envs < 1
    """
    dt = get(cfg, "environment.dt")
    if dt is not None and dt <= 0:
        raise ValueError(f"environment.dt must be > 0, got {dt}")

    num_envs = get(cfg, "environment.num_parallel_envs")
    if num_envs is not None and num_envs < 1:
        raise ValueError(f"environment.num_parallel_envs must be >= 1, got {num_envs}")

    def _check_speed_and_accel(d: dict, prefix: str = ""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _check_speed_and_accel(v, full_key)
            elif ("speed" in k or "accel" in k) and isinstance(v, (int, float)):
                if v <= 0:
                    raise ValueError(f"{full_key} must be > 0, got {v}")

    _check_speed_and_accel(cfg)

    decoy_fraction = get(cfg, "entities.missiles.decoy_fraction")
    if decoy_fraction is not None and (decoy_fraction < 0.0 or decoy_fraction > 1.0):
        raise ValueError(f"entities.missiles.decoy_fraction must be in [0, 1], got {decoy_fraction}")

    max_salvo_owners_k = get(cfg, "entities.missiles.max_salvo_owners_k")
    if max_salvo_owners_k is not None and max_salvo_owners_k not in (1, 2):
        raise ValueError(f"entities.missiles.max_salvo_owners_k must be 1 or 2, got {max_salvo_owners_k}")
