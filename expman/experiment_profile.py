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

from typing import Any, Dict, Iterable, List, Optional

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

from expman.config_utils import to_plain_dict, get_by_dot

def find_checkpoints(exp_dir: Path) -> List[str]:
    """Search for .ckpt / .pt / .pth files under common places and return sorted list (by mtime desc)."""
    patterns = ["**/*.ckpt", "**/*.pt", "**/*.pth"]
    found = []
    for pat in patterns:
        for p in exp_dir.glob(pat):
            if p.is_file():
                found.append(p)
    # sort newest first
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in found]


def find_wandb_dir(exp_dir: Path) -> Optional[str]:
    # wandb usually creates a "wandb" folder with run-id subfolders
    cand = exp_dir / "wandb"
    if cand.exists() and cand.is_dir():
        return str(cand)
    # sometimes logs are in "log" or "wandb/run-..."
    for p in exp_dir.glob("**/wandb*"):
        if p.is_dir():
            return str(p)
    return None


def git_commit_hash(cwd: Optional[Path] = None) -> Optional[str]:
    """Return current git commit hash if available; otherwise None."""
    try:
        cwd = cwd or Path.cwd()
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class ExperimentProfile:
    # human description
    description: str = ""

    # which keys to keep in the summary (dot notation supported)
    keep_keys: List[str] = field(default_factory=list)

    # how to generate the experiment folder name/id
    id_mode: str = "hash"  # one of 'sequential', 'hash', 'timestamp', 'uuid'
    hash_length: int = 8

    # do we snapshot some meta like git/python/torch versions
    include_meta: bool = True

    # optional tags to help search
    tags: List[str] = field(default_factory=list)

    # optional: list of extra files to include in id_card (relative globs)
    extra_files: List[str] = field(default_factory=list)

    # name of the id_card file to write
    id_card_name: str = "id_card.json"

    # Repo path

    @classmethod
    def from_profile_file(cls, p: str | Path) -> "ExperimentProfile": # ?? why annotations from future then ?
        p = Path(p).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"profile file not found: {p}")
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentProfile":
        return cls(**data)

    def _selected_summary(self, cfg: Any) -> Dict[str, Any]:
        cfgd = to_plain_dict(cfg)
        if not self.keep_keys:
            # if empty, return entire config
            return cfgd
        summary = {}
        for k in self.keep_keys:
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

    def _stable_hash(self, selected: Dict[str, Any]) -> str:
        j = json.dumps(selected, sort_keys=True, default=str)
        h = hashlib.sha1(j.encode("utf-8")).hexdigest()
        return h[: self.hash_length]

    def _next_sequential_name(self, root: Path, prefix="exp_", width=4) -> str:
        root = root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        existing = [p.name for p in root.iterdir() if p.is_dir() and re.match(rf"{re.escape(prefix)}\d+", p.name)]
        nums = []
        for name in existing:
            m = re.match(rf"{re.escape(prefix)}(\d+)", name)
            if m:
                nums.append(int(m.group(1)))
        next_num = max(nums) + 1 if nums else 1
        return f"{prefix}{next_num:0{width}d}"

    def _timestamp_name(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{ts}"

    def generate_exp_name(self, root: Path, selected_summary: Dict[str, Any]) -> str:
        root = root.expanduser()
        if self.id_mode == "sequential":
            return self._next_sequential_name(root)
        elif self.id_mode == "timestamp":
            return self._timestamp_name()
        elif self.id_mode == "hash":
            short = self._stable_hash(selected_summary)
            return f"exp_{short}"
        elif self.id_mode == "uuid":
            return f"exp_{uuid.uuid4().hex[: self.hash_length]}"
        else:
            raise ValueError(f"unknown id_mode: {self.id_mode}")

    def create_id_card(
        self,
        cfg: Any,
        exp_root: str | Path,
        exp_dir: Optional[str | Path] = None,
        copy_config_file: Optional[str | Path] = None,
        update_after: bool = True,
    ) -> Dict[str, Any]:
        """
        Create an id_card.json in the experiment directory.
        - cfg: hydra/DictConfig or dict
        - exp_root: root directory where experiments live (if exp_dir None, this function will pick/generate one)
        - exp_dir: optional explicit experiment directory path (if provided, we will use it)
        - copy_config_file: optional path to a config yaml that will be copied into exp_dir/config.yaml
        - update_after: if True, try to add artifacts (ckpt, wandb) immediately after (useful if called at the end of run)
        """
        exp_root = Path(exp_root).expanduser()
        selected = self._selected_summary(cfg)
        stable_hash = self._stable_hash(selected)

        # determine exp_dir (name + create)
        if exp_dir:
            exp_dir_path = Path(exp_dir).expanduser()
            exp_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            name = self.generate_exp_name(exp_root, selected)
            exp_dir_path = exp_root / name
            exp_dir_path.mkdir(parents=True, exist_ok=True)

        # optionally copy provided config file
        if copy_config_file:
            try:
                from shutil import copy2
                dst = exp_dir_path / "config_snapshot.yaml"
                copy2(str(copy_config_file), str(dst))
            except Exception:
                pass

        id_card = {
            "id": exp_dir_path.name,  # experiment directory name as the main ID
            "stable_hash": stable_hash,
            "created_at": datetime.datetime.now().isoformat(),
            "description": self.description,
            "profile": {
                "keep_keys": self.keep_keys,
                "id_mode": self.id_mode,
                "hash_length": self.hash_length,
                "tags": self.tags,
                "extra_files": self.extra_files,
            },
            "selected_fields": selected,
            "files": {},
            "meta": {},
        }

        if self.include_meta:
            id_card["meta"].update(
                {
                    "user": getpass.getuser(),
                    "host": socket.gethostname(),
                    "cwd": str(Path.cwd()),
                    "git_commit": git_commit_hash(Path.cwd()),
                    "python_version": platform.python_version(),
                }
            )
            # optional framework versions (best-effort)
            try:
                import importlib

                def _try_ver(name):
                    try:
                        mod = importlib.import_module(name)
                        return getattr(mod, "__version__", None)
                    except Exception:
                        return None

                id_card["meta"]["torch_version"] = _try_ver("torch")
                id_card["meta"]["lightning_version"] = _try_ver("pytorch_lightning") or _try_ver("lightning")
                id_card["meta"]["wandb_version"] = _try_ver("wandb")
            except Exception:
                pass

        # find common artifact places
        ckpts = find_checkpoints(exp_dir_path)
        wandb_dir = find_wandb_dir(exp_dir_path)
        if ckpts:
            id_card["files"]["checkpoints"] = ckpts
            id_card["files"]["best"] = ckpts[0] if ckpts else None
        if wandb_dir:
            id_card["files"]["wandb"] = wandb_dir

        # extra files globs
        extra_found = {}
        for pattern in self.extra_files:
            found = [str(p) for p in exp_dir_path.glob(pattern)]
            if found:
                extra_found[pattern] = found
        if extra_found:
            id_card["files"]["extra"] = extra_found

        # write id_card.json atomically
        tmp = exp_dir_path / (self.id_card_name + ".tmp")
        out = exp_dir_path / self.id_card_name
        with open(tmp, "w") as f:
            json.dump(id_card, f, indent=2, sort_keys=False, default=str)
        tmp.replace(out)

        if update_after:
            # re-run an update function to make sure files found after training are captured
            self.update_id_card_with_artifacts(exp_dir_path)

        return id_card

    def update_id_card_with_artifacts(self, exp_dir: str | Path) -> Dict[str, Any]:
        """Open id_card.json, update with discovered ckpts/wandb etc, write back."""
        exp_dir = Path(exp_dir).expanduser()
        id_card_path = exp_dir / self.id_card_name
        if not id_card_path.exists():
            raise FileNotFoundError(f"{id_card_path} not found")
        with open(id_card_path, "r") as f:
            id_card = json.load(f)

        # refresh artifact lists
        ckpts = find_checkpoints(exp_dir)
        if ckpts:
            id_card.setdefault("files", {})["checkpoints"] = ckpts
            id_card["files"]["best"] = ckpts[0]

        wandb_dir = find_wandb_dir(exp_dir)
        if wandb_dir:
            id_card.setdefault("files", {})["wandb"] = wandb_dir

        # update write
        tmp = id_card_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(id_card, f, indent=2, sort_keys=False, default=str)
        tmp.replace(id_card_path)
        return id_card

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


# small CLI for convenience (not a full CLI package)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create or update an experiment id_card.json using ExperimentProfile")
    parser.add_argument("--profile", type=str, required=False, help="YAML profile file (experiment.profile)")
    parser.add_argument("--cfg", type=str, required=False, help="Path to a config snapshot (yaml) to include")
    parser.add_argument("--exp-root", type=str, default="~/experiments", help="Root directory for experiments")
    parser.add_argument("--exp-dir", type=str, required=False, help="Use an explicit experiment directory")
    parser.add_argument("--update-only", action="store_true", help="If set, update an existing id_card.json instead of creating one")
    args = parser.parse_args()

    if args.update_only:
        if not args.exp_dir:
            raise SystemExit("update-only requires --exp-dir")
        # find a profile (use default)
        profile = ExperimentProfile()
        print("Updating id_card in", args.exp_dir)
        updated = profile.update_id_card_with_artifacts(args.exp_dir)
        print("Updated:", json.dumps(updated, indent=2)[:200], "...")
    else:
        profile = ExperimentProfile.from_profile_file(args.profile) if args.profile else ExperimentProfile()
        cfg = {}
        if args.cfg:
            with open(args.cfg, "r") as f:
                cfg = yaml.safe_load(f)
        card = profile.create_id_card(cfg, args.exp_root, exp_dir=args.exp_dir, copy_config_file=args.cfg)
        print("id_card created at", Path(args.exp_root).expanduser() / card["id"])
