"""
CAPS CLI — Command-line interface for the research framework.

Commands:
    caps init              — Initialize project directories
    caps doctor            — System health check
    caps models            — Display configured/available models
    caps dataset generate  — Generate synthetic benchmark
    caps experiment run    — Run an experiment
    caps experiment status — Show experiment status
    caps results summarize — Summarize results
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="caps",
    help="CAPS — Communication-Adaptive Policy System",
    add_completion=False,
)
console = Console()

# Sub-commands
dataset_app = typer.Typer(help="Dataset management")
experiment_app = typer.Typer(help="Experiment management")
results_app = typer.Typer(help="Results analysis")
research_app = typer.Typer(help="Research pipeline (synthetic_v2 benchmark)")
app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")
app.add_typer(results_app, name="results")
app.add_typer(research_app, name="research")


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------

@app.command()
def init():
    """Initialize project directories and configuration."""
    dirs = [
        "data/synthetic",
        "data/processed",
        "experiments/results",
        "experiments/checkpoints",
        "experiments/configs",
        "docs",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        console.print(f"  ✓ Created {d}/")

    # Create .env.example if not exists
    if not Path(".env").exists():
        Path(".env").write_text("GROQ_API_KEY=your_key_here\n")
        console.print("  ✓ Created .env (add your Groq API key)")

    console.print("\n[green]CAPS initialized successfully.[/green]")


@app.command()
def doctor():
    """System health check."""
    console.print("\n[bold]CAPS System Check[/bold]\n")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    status = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(f"  {status} Python {py_ver} (need ≥3.11)")

    # Groq API key
    from caps.config import get_groq_api_key
    key = get_groq_api_key()
    if key and key != "your_key_here" and key != "your_groq_api_key_here":
        console.print(f"  [green]✓[/green] Groq API key found ({key[:8]}...)")
    else:
        console.print(f"  [yellow]![/yellow] Groq API key not configured (mock mode only)")

    # sentence-transformers
    try:
        import sentence_transformers
        console.print(f"  [green]✓[/green] sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        console.print(f"  [red]✗[/red] sentence-transformers not installed")

    # Data directories
    for d in ["data", "experiments/results"]:
        exists = Path(d).exists()
        status = "[green]✓[/green]" if exists else "[yellow]![/yellow]"
        console.print(f"  {status} {d}/ {'exists' if exists else 'missing (run: caps init)'}")

    console.print()


@app.command()
def models(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file"),
):
    """Display configured and available models."""
    from caps.config import load_config, get_groq_api_keys

    cfg = load_config(config)

    table = Table(title="Configured Models")
    table.add_column("Role", style="cyan")
    table.add_column("Model ID", style="white")
    table.add_row("Planner", cfg.models.planner)
    table.add_row("Receiver", cfg.models.receiver)
    table.add_row("Evaluator", cfg.models.evaluator)
    console.print(table)

    # Try to list available models from Groq
    keys = get_groq_api_keys()
    if keys and all(k not in ("your_key_here", "your_groq_api_key_here") for k in keys):
        try:
            from caps.llm.groq_client import GroqProvider
            from caps.llm.model_registry import ModelRegistry

            provider = GroqProvider(api_key=keys)
            registry = ModelRegistry(provider)
            console.print(registry.display({
                "planner": cfg.models.planner,
                "receiver": cfg.models.receiver,
                "evaluator": cfg.models.evaluator,
            }))
        except Exception as e:
            console.print(f"[yellow]Could not fetch models: {e}[/yellow]")
    else:
        console.print("\n[yellow]Set GROQ_API_KEY to check model availability.[/yellow]")


# ---------------------------------------------------------------------------
# Dataset commands
# ---------------------------------------------------------------------------

@dataset_app.command("generate")
def dataset_generate(
    size: int = typer.Option(50, "--size", "-n", help="Number of tasks"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    output: str = typer.Option("data/synthetic/benchmark.jsonl", "--output", "-o"),
    families: Optional[str] = typer.Option(None, "--families", help="Comma-separated families"),
):
    """Generate a synthetic benchmark dataset."""
    from caps.datasets.generator import generate_dataset, save_dataset

    family_list = families.split(",") if families else None
    tasks = generate_dataset(families=family_list, size=size, seed=seed)
    save_dataset(tasks, output)

    console.print(f"\n[green]Generated {len(tasks)} tasks → {output}[/green]")

    # Summary by category
    from collections import Counter
    cats = Counter(t["category"] for t in tasks)
    for cat, count in sorted(cats.items()):
        console.print(f"  {cat}: {count}")


@dataset_app.command("validate")
def dataset_validate(
    path: str = typer.Argument(..., help="Path to JSONL dataset"),
):
    """Validate a dataset file."""
    from caps.datasets.generator import load_dataset

    tasks = load_dataset(path)
    console.print(f"\nLoaded {len(tasks)} tasks from {path}")

    errors = 0
    for i, task in enumerate(tasks):
        if "task_id" not in task:
            console.print(f"  [red]Task {i}: missing task_id[/red]")
            errors += 1
        if "cius" not in task or not task["cius"]:
            console.print(f"  [red]Task {i}: missing CIUs[/red]")
            errors += 1
        if "ground_truth" not in task:
            console.print(f"  [red]Task {i}: missing ground_truth[/red]")
            errors += 1

    if errors == 0:
        console.print("[green]✓ Dataset validation passed.[/green]")
    else:
        console.print(f"[red]✗ {errors} validation errors found.[/red]")


# ---------------------------------------------------------------------------
# Experiment commands
# ---------------------------------------------------------------------------

@experiment_app.command("run")
def experiment_run(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run an experiment."""
    from caps.config import load_config, get_groq_api_keys
    from caps.experiments.runner import ExperimentRunner

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config with CLI overrides
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if verbose:
        overrides["verbose"] = True

    cfg = load_config(config, cli_overrides=overrides)
    console.print(f"\n[bold]CAPS Experiment: {cfg.name}[/bold]")
    console.print(f"  Seed: {cfg.seed}")
    console.print(f"  Models: {cfg.models.planner} / {cfg.models.receiver}")
    console.print(f"  Policy: {cfg.policy.algorithm} (α={cfg.policy.alpha})")
    console.print(f"  Dataset: {cfg.dataset.name} ({cfg.dataset.train_size + cfg.dataset.validation_size + cfg.dataset.test_size} tasks)")
    console.print()

    # Create LLM provider
    if cfg.models.planner == "mock":
        from caps.llm.mock_client import MockLLMProvider
        llm = MockLLMProvider()
        console.print("[yellow]Using mock LLM provider (no API calls)[/yellow]\n")
    else:
        from caps.config import get_groq_api_keys, get_openrouter_api_keys
        or_keys = get_openrouter_api_keys()
        
        if or_keys and not all(k in ("your_key_here", "your_groq_api_key_here") for k in or_keys):
            console.print("[green]Using OpenRouter API[/green]\n")
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(
                api_key=or_keys, 
                default_model=cfg.models.planner,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            keys = get_groq_api_keys()
            if not keys or all(k in ("your_key_here", "your_groq_api_key_here") for k in keys):
                console.print("[red]Error: GROQ_API_KEY or OPEN_ROUTER_API_KEY not configured.[/red]")
                console.print("Set it in .env or environment, or use mock config.")
                raise typer.Exit(1)
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(api_key=keys, default_model=cfg.models.planner)

    # Run
    runner = ExperimentRunner(config=cfg, llm_provider=llm)
    results = runner.run()

    # Display results
    console.print("\n[bold]Results Summary[/bold]\n")
    table = Table()
    table.add_column("Policy", style="cyan")
    table.add_column("Success Rate", justify="right")
    table.add_column("Mean Tokens", justify="right")
    table.add_column("Clarification Rate", justify="right")

    for policy_name, metrics in results.items():
        table.add_row(
            policy_name,
            f"{metrics.get('success_rate', 0):.3f}",
            f"{metrics.get('mean_tokens', 0):.0f}",
            f"{metrics.get('clarification_rate', 0):.3f}",
        )

    console.print(table)
    console.print(f"\n[green]Results saved to {cfg.database_path}[/green]")


@experiment_app.command("status")
def experiment_status():
    """Show status of all experiments."""
    from caps.config import load_config
    from caps.storage.database import ExperimentDB

    cfg = load_config()
    db = ExperimentDB(cfg.database_path)
    experiments = db.get_all_experiments()

    if not experiments:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Policy")
    table.add_column("Status")
    table.add_column("Episodes")
    table.add_column("Created")

    for exp in experiments:
        table.add_row(
            exp.experiment_id[:20],
            exp.name[:30],
            exp.policy_name,
            exp.status,
            f"{exp.completed_episodes}/{exp.total_episodes}",
            str(exp.created_at)[:19] if exp.created_at else "",
        )

    console.print(table)


@experiment_app.command("estimate-cost")
def experiment_estimate_cost(
    config: str = typer.Option(
        "configs/caps_synthetic.yaml",
        "--config", "-c",
    ),
):
    """Estimate token usage for an experiment configuration."""
    from caps.config import load_config

    cfg = load_config(config)
    total_tasks = cfg.dataset.train_size + cfg.dataset.validation_size + cfg.dataset.test_size
    num_policies = 5  # Full, Random, Heuristic, LLMPlanner, CAPS
    avg_cius_per_task = 6
    avg_tokens_per_ciu = 50

    # Estimates
    est_receiver_calls = total_tasks * num_policies
    est_planner_calls = total_tasks  # LLMPlanner baseline only
    est_input_per_call = avg_cius_per_task * avg_tokens_per_ciu
    est_output_per_call = 200

    est_total = (est_receiver_calls + est_planner_calls) * (est_input_per_call + est_output_per_call)
    counterfactual_overhead = 0
    if cfg.evaluator.counterfactual:
        counterfactual_overhead = total_tasks * cfg.evaluator.max_counterfactuals_per_task * (est_input_per_call + est_output_per_call)
    est_total += counterfactual_overhead

    console.print(f"\n[bold]Cost Estimate: {cfg.name}[/bold]\n")
    console.print(f"  Total tasks:              {total_tasks}")
    console.print(f"  Policies:                 {num_policies}")
    console.print(f"  Est. LLM calls:           {est_receiver_calls + est_planner_calls}")
    console.print(f"  Est. counterfactual calls: {total_tasks * cfg.evaluator.max_counterfactuals_per_task if cfg.evaluator.counterfactual else 0}")
    console.print(f"  Est. input tokens/call:   {est_input_per_call}")
    console.print(f"  Est. output tokens/call:  {est_output_per_call}")
    console.print(f"  [bold]Est. total tokens:       {est_total:,}[/bold]")
    console.print(f"  Configured daily budget:  {cfg.budget.max_daily_tokens:,}")
    days = max(1, est_total // cfg.budget.max_daily_tokens)
    console.print(f"  Est. days needed:         {days}")


# ---------------------------------------------------------------------------
# Results commands
# ---------------------------------------------------------------------------

@results_app.command("summarize")
def results_summarize(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Summarize experiment results."""
    from caps.config import load_config
    from caps.storage.database import ExperimentDB

    cfg = load_config(config)
    db = ExperimentDB(cfg.database_path)
    experiments = db.get_all_experiments()

    for exp in experiments:
        metrics = db.get_metrics(exp.experiment_id)
        if metrics:
            console.print(f"\n[cyan]{exp.name}[/cyan] ({exp.policy_name})")
            for name, value in metrics.items():
                console.print(f"  {name}: {value:.4f}")


@results_app.command("export")
def results_export(
    format: str = typer.Option("csv", "--format", "-f", help="csv|markdown|latex"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    output: str = typer.Option("experiments/results/summary", "--output", "-o"),
):
    """Export results in various formats."""
    import pandas as pd
    from caps.config import load_config
    from caps.storage.database import ExperimentDB

    cfg = load_config(config)
    db = ExperimentDB(cfg.database_path)

    experiments = db.get_all_experiments()
    rows = []
    for exp in experiments:
        metrics = db.get_metrics(exp.experiment_id)
        row = {"policy": exp.policy_name, "seed": exp.seed, **metrics}
        rows.append(row)

    if not rows:
        console.print("[yellow]No results to export.[/yellow]")
        return

    df = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        path = f"{output}.csv"
        df.to_csv(path, index=False)
    elif format == "markdown":
        path = f"{output}.md"
        Path(path).write_text(df.to_markdown(index=False))
    elif format == "latex":
        path = f"{output}.tex"
        Path(path).write_text(df.to_latex(index=False))
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
        return

    console.print(f"[green]Exported to {path}[/green]")


# ---------------------------------------------------------------------------
# Research commands — synthetic_v2 pipeline
# ---------------------------------------------------------------------------

@research_app.command("run")
def research_run(
    config: str = typer.Option(
        "configs/caps_research_mock.yaml", "--config", "-c",
        help="Path to research YAML config",
    ),
    resume: Optional[str] = typer.Option(
        None, "--resume", "-r",
        help="Resume an existing run_id (e.g., research_1a2b3c4d)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the research pipeline (synthetic_v2 benchmark, paired train/val/test)."""
    import yaml
    from caps.llm.mock_client import MockLLMProvider
    from caps.research.runner import ResearchRunner

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        raise typer.Exit(1)

    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # LLM provider
    model_name = raw.get("models", {}).get("planner", "mock")
    if model_name == "mock":
        llm = MockLLMProvider()
        console.print("[yellow]Using mock LLM provider[/yellow]")
    else:
        from caps.config import get_groq_api_keys, get_openrouter_api_keys
        or_keys = get_openrouter_api_keys()
        if or_keys and not all(k in ("your_key_here",) for k in or_keys):
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(api_key=or_keys, default_model=model_name,
                               base_url="https://openrouter.ai/api/v1")
        else:
            keys = get_groq_api_keys()
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(api_key=keys, default_model=model_name)

    dataset_cfg = raw.get("dataset", {})
    runner = ResearchRunner(
        llm_provider=llm,
        dataset_size=dataset_cfg.get("size", 35),
        dataset_seed=dataset_cfg.get("seed", 1),
        budget_fractions=raw.get("budget_fractions", [1.0, 0.5, 0.3]),
        seeds=raw.get("seeds", [1, 2, 3]),
        methods=raw.get("methods", None),
        linucb_alpha=raw.get("policy", {}).get("alpha", 1.0),
        results_dir=raw.get("results_dir", "research/results"),
        checkpoint_dir=raw.get("checkpoint_dir", "research/checkpoints"),
        manifest_dir=raw.get("manifest_dir", "research/manifests"),
        data_dir=raw.get("data_dir", "data/research"),
        embedding_model=raw.get("embedding_model", "all-MiniLM-L6-v2"),
        verbose=verbose,
        resume_run_id=resume,
    )

    console.print(f"\n[bold]CAPS Research Run: {runner.run_id}[/bold]")
    console.print(f"  Config: {cfg_path}")
    console.print(f"  Methods: {', '.join(runner.methods)}")
    console.print(f"  Budgets: {runner.budget_fractions}")
    console.print(f"  Seeds: {runner.seeds}")
    console.print()

    output = runner.run()

    console.print(f"\n[green]Research run complete![/green]")
    console.print(f"  Run ID:       {output['run_id']}")
    console.print(f"  Episodes CSV: {output['episodes_csv']}")
    console.print(f"  Manifest:     {output['manifest_path']}")
    console.print()

    # Print quick summary
    from caps.research.analysis import ResearchAnalysis
    try:
        ra = ResearchAnalysis(output["episodes_csv"])
        ra.print_summary()
    except Exception as e:
        console.print(f"[yellow]Summary skipped: {e}[/yellow]")


@research_app.command("run-pilot")
def research_run_pilot(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a small-scale real-LLM pilot to validate the pipeline."""
    from caps.research.runner import ResearchRunner
    from caps.config import get_groq_api_keys, get_openrouter_api_keys

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console.print("\n[bold]CAPS Real-LLM Pilot Run[/bold]\n")

    # Force real LLM (use an available model from the registry)
    model_name = "qwen/qwen3.6-27b"
    or_keys = get_openrouter_api_keys()
    
    try:
        if or_keys and not all(k in ("your_key_here", "your_groq_api_key_here") for k in or_keys):
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(api_key=or_keys, default_model="meta-llama/llama-3.1-8b-instruct",
                               base_url="https://openrouter.ai/api/v1")
            console.print("[green]Using OpenRouter API (meta-llama/llama-3.1-8b-instruct)[/green]")
        else:
            keys = get_groq_api_keys()
            if not keys or all(k in ("your_key_here", "your_groq_api_key_here") for k in keys):
                console.print("[red]Error: GROQ_API_KEY not configured. Pilot requires a real LLM.[/red]")
                raise typer.Exit(1)
            from caps.llm.groq_client import GroqProvider
            llm = GroqProvider(api_key=keys, default_model=model_name)
            console.print("[green]Using Groq API[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing LLM: {e}[/red]")
        raise typer.Exit(1)

    # Pilot configuration
    dataset_size = 10
    budget_fractions = [1.0, 0.5]
    seeds = [1]
    methods = [
        "FullContext-Unbounded",
    ]

    console.print(f"  Model:   {model_name}")
    console.print(f"  Tasks:   {dataset_size}")
    console.print(f"  Budgets: {budget_fractions}")
    console.print(f"  Seeds:   {seeds}")
    console.print(f"  Methods: {', '.join(methods)}\n")

    runner = ResearchRunner(
        llm_provider=llm,
        dataset_size=dataset_size,
        dataset_seed=42, # fixed pilot seed
        budget_fractions=budget_fractions,
        seeds=seeds,
        methods=methods,
        results_dir="research_pilot",
        checkpoint_dir="research_pilot/checkpoints",
        manifest_dir="research_pilot",
        data_dir="research_pilot/data",
        verbose=verbose,
    )

    output = runner.run()

    console.print(f"\n[green]Pilot run complete![/green]")
    console.print(f"  Run ID:       {output['run_id']}")
    console.print(f"  Episodes CSV: {output['episodes_csv']}")
    console.print(f"  Manifest:     {output['manifest_path']}")
    console.print()


@research_app.command("validate")
def research_validate(
    episodes_csv: str = typer.Argument(..., help="Path to *_episodes.csv"),
    splits_json: str = typer.Option(
        "data/research/splits_1.json", "--splits", "-s",
    ),
    dataset_jsonl: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Full dataset JSONL for leakage check",
    ),
):
    """Validate a completed research run for data quality and correctness."""
    from caps.research.validation import ResearchValidator

    v = ResearchValidator()
    report = v.validate_all(
        episode_csv=episodes_csv,
        splits_path=splits_json,
        dataset_jsonl=dataset_jsonl,
    )
    v.print_report(report)

    if report["overall"] == "FAIL":
        raise typer.Exit(code=1)


@research_app.command("report")
def research_report(
    episodes_csv: str = typer.Argument(..., help="Path to *_episodes.csv"),
    output: str = typer.Option(
        "research/results/summary.md", "--output", "-o",
    ),
    budget_fraction: Optional[float] = typer.Option(
        None, "--budget", "-b", help="Filter to specific budget fraction",
    ),
):
    """Generate a Markdown summary table from a research episode CSV."""
    from caps.research.analysis import ResearchAnalysis

    ra = ResearchAnalysis(episodes_csv)
    ra.print_summary(budget_fraction=budget_fraction)
    ra.save_markdown(output, budget_fraction=budget_fraction)

    csv_out = output.replace(".md", ".csv")
    ra.save_summary_csv(csv_out)

    console.print(f"[green]Saved:[/green] {output}")
    console.print(f"[green]Saved:[/green] {csv_out}")


# ---------------------------------------------------------------------------
# DB inspection
# ---------------------------------------------------------------------------

@app.command("db")
def db_inspect(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Inspect the experiment database."""
    from caps.config import load_config
    from caps.storage.database import ExperimentDB

    cfg = load_config(config)
    db = ExperimentDB(cfg.database_path)

    experiments = db.get_all_experiments()
    console.print(f"\nDatabase: {cfg.database_path}")
    console.print(f"Experiments: {len(experiments)}")

    for exp in experiments:
        ep_count = db.count_episodes(exp.experiment_id)
        console.print(
            f"  [{exp.status}] {exp.experiment_id} — "
            f"{exp.policy_name} — {ep_count} episodes"
        )


if __name__ == "__main__":
    app()
