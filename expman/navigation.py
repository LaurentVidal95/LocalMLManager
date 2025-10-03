"""
All functions related to the navigation in the experiments root directory.
"""

# expman/listing.py
import json
from pathlib import Path
import pandas as pd

from typing import Dict, List, Optional

# Small utils for paths.
from expman.utils import clean_path
from expman.core import ExperimentProfile

# def list_experiments(
#         exp_root: str | Path = "~/experiments",
#         id_card_name: str = "id_card.json"
# ) -> pd.DataFrame:
#     exp_root = clean_path(exp_root)
#     exps = []
#     for exp_dir in sorted(exp_root.glob("exp_*")):
#         card_file = exp_dir / id_card_name
#         if card_file.exists():
#             with open(card_file) as f:
#                 exps.append(json.load(f))
#     if not exps:
#         return pd.DataFrame()
#     df = pd.DataFrame(exps)
#     return df

def list_experiments(
        exp_root: str | Path = "~/experiments",
        id_card_name: str = "id_card.json",
        filters=None) -> pd.DataFrame:
    exp_root = clean_path(exp_root)
    exps = []
    for exp_dir in sorted(exp_root.glob("exp_*")):
        card_file = exp_dir / id_card_name
        if card_file.exists():
            with open(card_file) as f:
                card = json.load(f)
                # flatten summary row
                row = {
                    "id": card["id"],
                    "created_at": card.get("created_at"),
                    "description": card.get("description", ""),
                }
                # flatten selected keys
                for k, v in (card.get("config_summary") or {}).items():
                    row[k] = v
                row["tags"] = str(card.get("tags")).strip("[]")

                exps.append(row)

    df = pd.DataFrame(exps)
    if df.empty:
        return df

    # Apply filters if provided
    if filters:
        for key, val in filters.items():
            if key in df.columns:
                df = df[df[key] == val]

    return df.reset_index(drop=True)

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
    ckpts = [str(ckpt) for ckpt in found]
    return ckpts if ckpts else None


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

