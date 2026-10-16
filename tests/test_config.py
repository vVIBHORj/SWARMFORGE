from pathlib import Path
import pytest

from src.config import get, load_config, validate_config


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_all_configs():
    default_cfg = load_config(CONFIGS_DIR / "default_sim.yaml")
    train_cfg = load_config(CONFIGS_DIR / "train_mappo_rtx3050.yaml")
    stress_cfg = load_config(CONFIGS_DIR / "stress_test_300v500.yaml")

    assert isinstance(default_cfg, dict)
    assert isinstance(train_cfg, dict)
    assert isinstance(stress_cfg, dict)

    validate_config(default_cfg)
    validate_config(train_cfg)
    validate_config(stress_cfg)


def test_extends_merge():
    train_cfg = load_config(CONFIGS_DIR / "train_mappo_rtx3050.yaml")
    # Overridden value
    assert get(train_cfg, "environment.num_parallel_envs") == 16
    # Inherited value from default_sim.yaml
    assert get(train_cfg, "entities.drones.max_speed") == 350.0
    assert get(train_cfg, "training.algorithm") == "attention_mappo_ctde"

    stress_cfg = load_config(CONFIGS_DIR / "stress_test_300v500.yaml")
    assert get(stress_cfg, "environment.num_parallel_envs") == 1
    assert get(stress_cfg, "entities.drones.num_interceptors") == 240
    assert get(stress_cfg, "entities.drones.num_scouts") == 60
    assert get(stress_cfg, "entities.missiles.num_missiles") == 500
    assert get(stress_cfg, "entities.drones.max_speed") == 350.0


def test_validate_config_rejects_bad_dt():
    cfg = load_config(CONFIGS_DIR / "default_sim.yaml")
    cfg["environment"]["dt"] = -0.01
    with pytest.raises(ValueError, match="dt"):
        validate_config(cfg)

    cfg["environment"]["dt"] = 0.0
    with pytest.raises(ValueError, match="dt"):
        validate_config(cfg)


def test_validate_config_rejects_bad_decoy_fraction():
    cfg = load_config(CONFIGS_DIR / "default_sim.yaml")
    cfg["entities"]["missiles"]["decoy_fraction"] = 1.5
    with pytest.raises(ValueError, match="decoy_fraction"):
        validate_config(cfg)

    cfg["entities"]["missiles"]["decoy_fraction"] = -0.1
    with pytest.raises(ValueError, match="decoy_fraction"):
        validate_config(cfg)
