# expman/cli.py
import typer
import json
import yaml
from typing import List
from pathlib import Path

from expman.core import ExperimentProfile, create_experiment
from expman.navigation import list_experiments

app = typer.Typer(help="A simple ML experiment manager in full CLI")

@app.command()
def create(
    profile: str = typer.Option(..., "--profile", help="Path to experiment.profile YAML"),
    cfg: str = typer.Option(None, "--cfg", help="Path to Hydra config snapshot"),
    input_dir: str = typer.Option(..., "--input-dir", help="Original run directory"),
    exp_root: str = typer.Option("~/experiments", "--exp-root", help="Root directory for experiments"),
):
    """Wrap an existing run into an experiment folder with id_card.json"""
    prof = ExperimentProfile(**yaml.safe_load(open(profile)))
    cfg_dict = {}
    if cfg:
        cfg_dict = yaml.safe_load(open(cfg))
    card = create_experiment(prof, cfg_dict, exp_root, input_dir)
    typer.echo(f"✅ Created {card['id']} at {Path(exp_root).expanduser() / card['id']}")

@app.command()
def ls(
    exp_root: str = "~/experiments",
    filter: List[str] = typer.Option(None, "--filter", help="Filter like key=value"),
):
    """List experiments (with optional filters)."""
    filters = {}
    if filter:
        for f in filter:
            if "=" not in f:
                raise typer.BadParameter("Filter must be key=value")
            k, v = f.split("=", 1)
            filters[k] = v

    df = list_experiments(exp_root, filters=filters)
    if df.empty:
        typer.echo("No experiments found.")
    else:
        typer.echo(df.to_string(index=False))


@app.command()
def ckpt(exp_id: str, exp_root: str = "~/experiments", ckpt_type: str = "best"):
    """Get checkpoint path from an experiment"""
    exp_dir = Path(exp_root).expanduser() / exp_id
    card_file = exp_dir / "id_card.json"
    if not card_file.exists():
        typer.echo("No id_card.json found")
        raise typer.Exit(code=1)
    card = json.load(open(card_file))
    ckpt_dir = Path(card["files"]["checkpoints"]) if card["files"]["checkpoints"] else None
    if not ckpt_dir or not ckpt_dir.exists():
        typer.echo("No checkpoints found")
    else:
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if ckpt_type == "last":
            typer.echo(str(ckpt_dir / "last.ckpt"))
        elif ckpts:
            typer.echo(str(ckpts[0]))
        else:
            typer.echo("No checkpoints found")

@app.command()
def inspect(exp_id: str, exp_root: str = "~/experiments"):
    """Show id_card.json"""
    card_file = Path(exp_root).expanduser() / exp_id / "id_card.json"
    typer.echo(json.dumps(json.load(open(card_file)), indent=2))
