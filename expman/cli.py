# expman/cli.py
import typer
import json
import yaml
from typing import List
from pathlib import Path
from tabulate import tabulate

from expman.core import ExperimentProfile, create_experiment
from expman.navigation import list_experiments, experiment_summary

app = typer.Typer(help="A simple ML experiment manager in full CLI")

@app.command()
def create(
    profile: str = typer.Option(..., "--profile", help="Path to experiment.profile YAML"),
    cfg: str = typer.Option(None, "--cfg", help="Path to Hydra config snapshot"),
    input_dir: str = typer.Option(..., "--input-dir", help="Original run directory"),
    exp_root: str = typer.Option("~/experiments", "--exp-root",\
                                 help="Root directory for experiments"),
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
    exp_root: str = typer.Argument("~", help="Path to experiment root"),
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
        typer.echo(tabulate(df, headers="keys", tablefmt="github", showindex=False))

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
def inspect_id_card(exp_id: str, exp_root: str = "~/experiments"):
    """Show id_card.json"""
    full_id = "exp_"+exp_id
    card_file = Path(exp_root).expanduser() / full_id / "id_card.json"
    typer.echo(json.dumps(json.load(open(card_file)), indent=2))

@app.command()
def show(
    exp_id: str = typer.Argument(..., help="Experiment ID (e.g. 03a48559)"),
    exp_root: str = typer.Option("~/experiments", "--exp-root", help="Root directory"),
    id_card_name=typer.Option("id_card.json", "--id-card-name", help="Name of the id file"),    
):
    """Show details of a single experiment (summary + checkpoints)."""
    root = Path(exp_root).expanduser()
    exp_id = "exp_" + exp_id
    exp_dir = root / exp_id

    # --- a) summary ---
    summary = experiment_summary(exp_dir, id_card_name)
    if summary is None:
        typer.echo(f"No id_card.json found for {exp_id} in {exp_root}")
        raise typer.Exit(1)

    typer.echo("Summary:")
    typer.echo(tabulate([summary], headers="keys", tablefmt="github"))

    # --- b) checkpoints ---
    ckpt_dir = exp_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = []
        for ckpt in sorted(ckpt_dir.glob("*.ckpt")):
            label = "last" if "last" in ckpt.name else "best"
            checkpoints.append({"file": ckpt.name, "label": label})
        typer.echo("\nCheckpoints:")
        typer.echo(tabulate(checkpoints, headers="keys", tablefmt="github"))
    else:
        typer.echo("\nNo checkpoints found.")
