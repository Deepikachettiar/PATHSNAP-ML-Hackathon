"""Configuration loading. Paths are resolved relative to the repo root."""
import os
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    cfg_path = Path(os.environ.get("PATHSNAP_CONFIG", REPO_ROOT / "configs" / "config.yaml"))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = os.environ.get("PATHSNAP_MODE", cfg.get("mode", "dev")).lower()
    cfg["engine"] = os.environ.get("PATHSNAP_ENGINE", cfg.get("engine", "auto")).lower()
    roots = cfg.setdefault("data_roots", {})
    if os.environ.get("PATHSNAP_DEV_ROOT"):
        roots["dev"] = os.environ["PATHSNAP_DEV_ROOT"]
    if os.environ.get("PATHSNAP_FINAL_ROOT"):
        roots["final"] = os.environ["PATHSNAP_FINAL_ROOT"]
    if cfg["mode"] not in ("dev", "final"):
        raise ValueError("mode must be 'dev' or 'final'")
    root = roots.get(cfg["mode"], "")
    if not root:
        raise ValueError(f"No data root configured for mode '{cfg['mode']}'. "
                         f"Set data_roots.{cfg['mode']} in configs/config.yaml or the env variable.")
    p = Path(root)
    cfg["data_root"] = str(p if p.is_absolute() else REPO_ROOT / p)
    return cfg
