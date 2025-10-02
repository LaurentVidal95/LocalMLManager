import json
import yaml
from typing import Any, Dict

# Try to support OmegaConf DictConfig transparently
try:
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError
    _HAS_OMEGACONF = True
except Exception:
    OmegaConf = None
    _HAS_OMEGACONF = False

# OMEGACONF MANAGMENT
def to_plain_dict(cfg: Any) -> Dict[str, Any]:
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
            return {k: to_plain_dict(v) for k, v in vars(cfg).items()}
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
