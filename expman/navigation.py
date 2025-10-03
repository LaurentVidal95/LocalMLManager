"""
All functions related to the navigation in the experiments root directory.
"""
from pathlib import Path
from typing import Dict, List, Optional

import json
import time
from datetime import datetime
import pandas as pd

from expman.utils import clean_path
from expman.core import ExperimentProfile

def nice_time(time_str: str, format: str = "%d/%m/%y - %H:%M") -> str:
    if time_str:
        try:
            dt = datetime.fromisoformat(time_str)
            return dt.strftime(format)
        except Exception:
            return time_str  # fallback
    return "None"

def experiment_summary(exp_dir: str | Path,
                       id_card_name: str = "id_card.json",
                       ) -> List[str]:
    exp_dir = clean_path(exp_dir)
    card_file = exp_dir / id_card_name
    if card_file.exists():
        with open(card_file) as f:
            card = json.load(f)
            output = {
                "id": card["id"].split("_")[-1],
                "description": card.get("description", ""),
            }

            # Add time
            output["time"] = nice_time(card.get("created_at"))

            # Add selected keys from config
            for k, v in (card.get("config_summary") or {}).items():
                output[k] = v

            # Add tags
            tags = card.get("tags", [])
            if isinstance(tags, list):
                output["tags"] = ", ".join(tags)
            else:
                output["tags"] = str(tags)
                
            commit = (card.get("meta") or {}).get("commit")
            if commit:
                output["commit"] = commit[:7]
        return output
    return None

def list_experiments(
        exp_root: str | Path = "~/experiments",
        id_card_name: str = "id_card.json",
        filters=None) -> pd.DataFrame:
    """
    TODO: doc
    """
    exp_root = clean_path(exp_root)
    exps = []
    for exp_dir in sorted(exp_root.glob("exp_*")):
        exps.append(experiment_summary(exp_dir, id_card_name=id_card_name))

    df = pd.DataFrame(exps)
    if df.empty:
        return df

    # Apply filters if provided
    if filters:
       for key, val in filters.items():
            if key in df.columns:
                df = df[df[key] == val]

    return df.reset_index(drop=True)

# def list_checkpoints(exp_dir) -> List[str]:
#     """
#     Search for .ckpt / .pt / .pth files under
#     common places and return sorted list (by mtime desc).
#     """
#     patterns = ["**/*.ckpt", "**/*.pt", "**/*.pth"]
#     found = []
#     for pat in patterns:
#         for p in exp_dir.glob(pat):
#             if p.is_file():
#                 found.append(p)
#     # sort newest first
#     found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
#     ckpts = [str(ckpt) for ckpt in found]
#     return ckpts if ckpts else None


# def find_wandb_dir(exp_dir: Path) -> Optional[str]:
#     # wandb usually creates a "wandb" folder with run-id subfolders
#     cand = exp_dir / "wandb"
#     if cand.exists() and cand.is_dir():
#         return str(cand)
#     # sometimes logs are in "log" or "wandb/run-..."
#     for p in exp_dir.glob("**/wandb*"):
#         if p.is_dir():
#             return str(p)
#     return None

