# experiment_profile.py
"""
ExperimentProfile: create & update an ID card for an experiment
- supports reading profile files (YAML)
- selecting a subset of keys from a (Hydra) config (dot notation)
- generates unique experiment folder names (sequential / hash / timestamp / uuid)
- writes id_card.json with metadata + selected fields
- can update id_card with found checkpoints / wandb dir
"""

from __future__ import annotations
import json
import yaml

from dataclasses import dataclass, field, asdict

from typing import Any, Dict, Iterable, List, Literal, Optional

from pathlib import Path
import hashlib
import datetime
import uuid
import re
import os
import platform
import getpass
import socket
import subprocess
import time

from expman.utils import clean_path, config_summary

@dataclass
class ExperimentProfile:
    """
    description: human description
    keep_keys: which keys to keep in the summary (dot notation supported)
    id_mode: how to generate the experiment folder name/id.
    hash_length: lenght of hash id if id_mode=hash
    tags:  optional tags to help filtering experiments
    id_card_name: name of the id_card json file to write
    extra_files: default list of extra files to include in the experiment folder.
                 Those are also listed in the id_card.
    """
    description: str = ""
    keep_keys: List[str] = None
    id_mode: Literal["hash", "sequential", "timestamp"] = "hash"
    hash_length: int = 8
    id_card_name: str = "id_card.json"
    model_repo: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list) # LV: check if that's necessary
    default_extra_files: List[str] = field(default_factory=list)

    @classmethod
    def from_profile_file(cls, p: str | Path) -> "ExperimentProfile": # LV: why annotations from future then ?
        p = clean_path(p)
        if not p.exists():
            raise FileNotFoundError(f"profile file not found: {p}")
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentProfile":
        return cls(**data)

    def _hash(self) -> str:
        payload = {"desc": self.description, "time": time.time()}
        j = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(j.encode("utf-8")).hexdigest()[: self.hash_length]

    def next_id(self, root: Path) -> str:
        root = clean_path(root)
        root.mkdir(parents=True, exist_ok=True)

        if self.id_mode == "sequential":
            existing = [exp.name for exp in root.glob("exp_*") if exp.is_dir()]
            nums = [int(name.split("_")[1]) for name in existing if name.split("_")[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            return f"exp_{next_num:04d}"

        elif self.id_mode == "hash":
            return "exp_" + self._hash()

        elif self.id_mode == "timestamp":
            return "exp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            raise ValueError(f"unknown id_mode: {self.id_mode}")

    def get_git_info(self) -> Dict[str, str]:
        """Return commit, branch, and remote URL for a git repo."""
        info = {"commit": None, "branch": None, "remote": None}
        if self.model_repo is None:
            return None

        model_repo = clean_path(self.model_repo)
        if not (model_repo / ".git").exists():
            return info
        try:
            info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=model_repo
            ).decode().strip()
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=model_repo
            ).decode().strip()
            info["remote"] = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], cwd=model_repo
            ).decode().strip()
        except Exception:
            pass
        return info

    # def update_id_card_with_artifacts(self, exp_dir: str | Path) -> Dict[str, Any]:
    #     """Open id_card.json, update with discovered ckpts/wandb etc, write back."""
    #     exp_dir = Path(exp_dir).expanduser()
    #     id_card_path = exp_dir / self.id_card_name
    #     if not id_card_path.exists():
    #         raise FileNotFoundError(f"{id_card_path} not found")
    #     with open(id_card_path, "r") as f:
    #         id_card = json.load(f)

    #     # refresh artifact lists
    #     ckpts = find_checkpoints(exp_dir)
    #     if ckpts:
    #         id_card.setdefault("files", {})["checkpoints"] = ckpts
    #         id_card["files"]["best"] = ckpts[0]

    #     wandb_dir = find_wandb_dir(exp_dir)
    #     if wandb_dir:
    #         id_card.setdefault("files", {})["wandb"] = wandb_dir

    #     # update write
    #     tmp = id_card_path.with_suffix(".json.tmp")
    #     with open(tmp, "w") as f:
    #         json.dump(id_card, f, indent=2, sort_keys=False, default=str)
    #     tmp.replace(id_card_path)
    #     return id_card

def create_id_card(
        profile: ExperimentProfile,
        config: Dict | DictConfig,
        exp_dir: Optional[str | Path]
) -> Dict[str, Any]:
    """Create an id_card.json inside exp_dir (no copying of logs/checkpoints)."""
    summary = config_summary(config, profile.keep_keys)
    exp_dir = clean_path(exp_dir)
    # git metadata
    git_info = {}
    if profile.model_repo:
        git_info = profile.get_git_info()
        
    id_card = {
        "id": exp_dir.name,
        "created_at": datetime.datetime.now().isoformat(),
        "description": profile.description,
        "config_summary": summary,
        "tags": profile.tags,
        "meta": {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            **git_info,  # merge commit/branch/remote
        },
        "files": {
            "checkpoints": str(exp_dir / "checkpoints") if (exp_dir / "checkpoints").exists() else None,
            "logs": str(exp_dir / "wandb") if (exp_dir / "wandb").exists() else None,
            "config": str(exp_dir / ".hydra/config.yaml") if (exp_dir / ".hydra/config.yaml").exists() else None,
        },
    }
    
    with open(exp_dir / "id_card.json", "w") as f:
        json.dump(id_card, f, indent=2)
        
    return id_card

def create_experiment(
    profile: ExperimentProfile,
    cfg: Any,
    exp_root: str | Path,
    input_dir: str | Path,
    delete_input_dir: bool = False
) -> Dict[str, Any]:

    """Wrap an old run into an experiment directory and generate id_card.json."""
    exp_root = Path(exp_root).expanduser()
    input_dir = Path(input_dir).expanduser()

    exp_id = profile.next_id(exp_root)
    exp_dir = exp_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # copy artifacts if they exist
    import shutil
    for sub in ["checkpoints", "wandb", ".hydra"]:
        src = input_dir / sub
        if src.exists():
            shutil.copytree(src, exp_dir / sub)

    # create id card
    return create_id_card(profile, cfg, exp_dir)
