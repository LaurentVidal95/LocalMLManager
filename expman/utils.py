import json
import yaml
from typing import Any, Dict, List
from pathlib import Path

# Try to support OmegaConf DictConfig transparently
try:
    from omegaconf import OmegaConf, DictConfig
    from omegaconf.errors import InterpolationResolutionError
    _HAS_OMEGACONF = True
except Exception:
    OmegaConf = None
    _HAS_OMEGACONF = False

def clean_path(path: str | Path):
    if isinstance(path, Path): 
        return path
    else:
        return Path(path).expanduser()

# OMEGACONF MANAGMENT
def to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert dict, dataclass, omegaconf.DictConfig into a plain nested dict (resolve=True)."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if _HAS_OMEGACONF and OmegaConf.is_config(cfg):
        try:
            return OmegaConf.to_container(cfg, resolve=True)
        except InterpolationResolutionError:
            return OmegaConf.to_container(cfg, resolve=False)

    # fallback: try dataclass or object with __dict__
    try:
        if hasattr(cfg, "__dict__"):
            return {k: to_dict(v) for k, v in vars(cfg).items()}
    except Exception:
        pass
    # last resort: convert via json (only if serializable)
    try:
        return json.loads(json.dumps(cfg))
    except Exception:
        return {}

def get_by_dot(d: Dict[str, Any], key: str, default: Any = None):
    """Get nested value by dot notation; e.g. 'optimizer.lr'."""
    cur = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur

def config_summary(cfg: Dict | DictConfig, keep_keys: List[str] = None) -> Dict[str, Any]:
    cfgd = to_dict(cfg)
    if not keep_keys:
        # if empty, return entire config
        return cfgd
    summary = {}
    for k in keep_keys:
        v = get_by_dot(cfgd, k, default=None)
        # allow top-level keys too; if not found, try partial match (prefix)
        if v is None:
            # find nested keys starting with k
            for cand_key, cand_val in flatten_dict(cfgd).items():
                if cand_key.startswith(k):
                    # collect them in nested dict under k
                    summary.setdefault(k, {})[cand_key[len(k)+1:]] = cand_val
        else:
            summary[k] = v
    return summary

# helper: flatten nested dict to dot keys (used optionally)
def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
