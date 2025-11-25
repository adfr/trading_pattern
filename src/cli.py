"""Command-line interface for the IBKR AI Trading System."""

import sys
from datetime import datetime, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.config import Config
from .core.database import Database
from .core.logger import setup_logger

console = Console()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration."""
    try:
        config = Config(config_path)
        return config
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)


def connect_ibkr(config: Config):
    """Connect to IBKR and return client."""
    from .data.ibkr_client import IBKRClient

    console.print("[yellow]Connecting to IBKR TWS/Gateway...[/yellow]")

    try:
        client = IBKRClient(config)
        client.connect()
        console.print("[green]Connected to IBKR[/green]")
        return client
    except Exception as e:
        console.print(f"[red]Failed to connect to IBKR: {e}[/red]")
        console.print()
        console.print("[yellow]Make sure TWS or IB Gateway is running:[/yellow]")
        console.print("  1. Open TWS or IB Gateway")
        console.print("  2. Go to Configure → API → Settings")
        console.print("  3. Enable 'Enable ActiveX and Socket Clients'")
        console.print(f"  4. Set Socket port to {config.ibkr.port}")
        console.print("  5. Add 127.0.0.1 to Trusted IPs")
        console.print()
        return None


@click.group()
@click.option("--config", "-c", type=click.Path(), help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str]) -> None:
    """IBKR AI Trading System - AI-powered algorithmic trading.

    All data is sourced from IBKR to ensure consistency between
    backtesting and live trading.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option("--symbol", "-s", default="QQQ", help="Symbol to analyze")
@click.option("--timeframe", "-t", default="1min", help="Data timeframe")
@click.option("--conditions", default="", help="Market conditions to consider")
@click.option("--count", "-n", default=1, help="Number of patterns to generate")
@click.pass_context
def generate(
    ctx: click.Context,
    symbol: str,
    timeframe: str,
    conditions: str,
    count: int,
) -> None:
    """Generate trading patterns using Claude AI.

    Requires IBKR connection for market data.
    """
    config = get_config(ctx.obj.get("config_path"))
    setup_logger("trading", config.logging.level, config.logging.file)

    console.print(Panel(f"Generating {count} pattern(s) for [bold]{symbol}[/bold]"))
    console.print("[dim]Data source: IBKR (ensures backtest/live consistency)[/dim]")
    console.print()

    # Connect to IBKR
    ibkr_client = connect_ibkr(config)
    if not ibkr_client:
        console.print("[red]Cannot generate patterns without IBKR connection.[/red]")
        return

    try:
        from .ai.pattern_generator import PatternGenerator
        from .data.historical import HistoricalDataManager

        # Get historical data from IBKR
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading market data from IBKR...", total=None)

            data_manager = HistoricalDataManager(config, ibkr_client=ibkr_client)
            df = data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=datetime.now() - timedelta(days=7),
            )

            if df.empty:
                console.print("[red]No data available for pattern generation[/red]")
                return

            console.print(f"[green]Loaded {len(df)} bars from IBKR[/green]")

        # Generate patterns
        database = Database(config.database_path)
        generator = PatternGenerator(config, database=database)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating patterns with Claude...", total=count)

            patterns = []
            for i in range(count):
                pattern = generator.generate_pattern(
                    symbol=symbol,
                    market_data=df,
                    timeframe=timeframe,
                    additional_context=conditions,
                )
                patterns.append(pattern)
                progress.update(task, advance=1)

        # Display results
        for pattern in patterns:
            table = Table(title=f"Pattern: {pattern['name']}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("ID", pattern.get("id", "N/A"))
            table.add_row("Type", pattern.get("pattern_type", "N/A"))
            table.add_row("Description", pattern.get("description", "N/A")[:100])
            table.add_row("Conditions", str(len(pattern.get("detection", {}).get("conditions", []))))
            table.add_row("Status", pattern.get("status", "pending"))

            console.print(table)
            console.print()

        console.print(f"[green]Generated {len(patterns)} pattern(s) successfully![/green]")
        console.print("Run [bold]backtest[/bold] command to validate patterns.")

    except Exception as e:
        console.print(f"[red]Error generating patterns: {e}[/red]")
        raise
    finally:
        if ibkr_client:
            ibkr_client.disconnect()


@cli.command()
@click.option("--pattern-id", "-p", help="Specific pattern ID to backtest")
@click.option("--all", "backtest_all", is_flag=True, help="Backtest all pending patterns")
@click.option("--start", type=click.DateTime(), help="Start date")
@click.option("--end", type=click.DateTime(), help="End date")
@click.option("--walk-forward", is_flag=True, help="Run walk-forward optimization")
@click.pass_context
def backtest(
    ctx: click.Context,
    pattern_id: Optional[str],
    backtest_all: bool,
    start: Optional[datetime],
    end: Optional[datetime],
    walk_forward: bool,
) -> None:
    """Backtest trading patterns using IBKR data.

    Uses same data source as live trading to ensure consistency.
    """
    config = get_config(ctx.obj.get("config_path"))
    setup_logger("trading", config.logging.level, config.logging.file)

    database = Database(config.database_path)

    # Get patterns to backtest
    if pattern_id:
        pattern = database.get_pattern(pattern_id)
        if not pattern:
            console.print(f"[red]Pattern not found: {pattern_id}[/red]")
            return
        patterns = [pattern]
    elif backtest_all:
        patterns = database.get_patterns(status="pending")
        if not patterns:
            console.print("[yellow]No pending patterns to backtest[/yellow]")
            return
    else:
        console.print("[red]Specify --pattern-id or --all[/red]")
        return

    console.print(Panel(f"Backtesting {len(patterns)} pattern(s)"))
    console.print("[dim]Data source: IBKR (ensures backtest/live consistency)[/dim]")
    console.print()

    # Connect to IBKR
    ibkr_client = connect_ibkr(config)
    if not ibkr_client:
        console.print("[red]Cannot backtest without IBKR connection.[/red]")
        return

    try:
        from .backtest.engine import BacktestEngine
        from .backtest.optimizer import WalkForwardOptimizer
        from .data.historical import HistoricalDataManager

        backtest_engine = BacktestEngine(config, database)
        data_manager = HistoricalDataManager(config, ibkr_client=ibkr_client)

        results_table = Table(title="Backtest Results")
        results_table.add_column("Pattern", style="cyan")
        results_table.add_column("Trades", justify="right")
        results_table.add_column("Win Rate", justify="right")
        results_table.add_column("Profit Factor", justify="right")
        results_table.add_column("Sharpe", justify="right")
        results_table.add_column("Max DD", justify="right")
        results_table.add_column("Result", justify="center")

        for pattern in patterns:
            symbol = pattern.get("symbol", "QQQ")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"Backtesting {pattern['name']}...", total=None)

                # Get data from IBKR
                df = data_manager.get_data(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    timeframe=pattern.get("timeframe", "1min"),
                )

                if df.empty:
                    console.print(f"[yellow]No IBKR data for {pattern['name']}[/yellow]")
                    continue

                # Run backtest
                if walk_forward:
                    optimizer = WalkForwardOptimizer(config, backtest_engine)
                    wf_result = optimizer.run(pattern, df)
                    result = wf_result
                    metrics = wf_result.out_of_sample_metrics
                    passed = wf_result.passed
                else:
                    result = backtest_engine.run(pattern, df, start, end)
                    metrics = result.metrics
                    passed = result.passed

                # Update pattern status
                new_status = "backtested" if passed else "failed"
                database.update_pattern_status(
                    pattern["id"],
                    new_status,
                    result.to_dict() if hasattr(result, "to_dict") else {},
                )

            # Add to results table
            if metrics:
                results_table.add_row(
                    pattern["name"][:30],
                    str(metrics.total_trades),
                    f"{metrics.win_rate:.1%}",
                    f"{metrics.profit_factor:.2f}",
                    f"{metrics.sharpe_ratio:.2f}",
                    f"{metrics.max_drawdown:.1%}",
                    "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
                )
            else:
                results_table.add_row(
                    pattern["name"][:30],
                    "0", "N/A", "N/A", "N/A", "N/A",
                    "[red]FAIL[/red]",
                )

        console.print(results_table)

    finally:
        if ibkr_client:
            ibkr_client.disconnect()


@cli.command()
@click.option("--pattern-id", "-p", required=True, help="Pattern ID to deploy")
@click.pass_context
def deploy(ctx: click.Context, pattern_id: str) -> None:
    """Deploy a pattern for live trading."""
    config = get_config(ctx.obj.get("config_path"))
    database = Database(config.database_path)

    pattern = database.get_pattern(pattern_id)
    if not pattern:
        console.print(f"[red]Pattern not found: {pattern_id}[/red]")
        return

    if not pattern.get("passed_backtest"):
        console.print("[red]Pattern has not passed backtesting![/red]")
        console.print("Run backtest first before deploying.")
        return

    database.update_pattern_status(pattern_id, "deployed")
    console.print(f"[green]Pattern '{pattern['name']}' deployed successfully![/green]")


@cli.command()
@click.option("--mode", "-m", type=click.Choice(["paper", "live"]), default="paper")
@click.pass_context
def trade(ctx: click.Context, mode: str) -> None:
    """Start live trading.

    Connects to IBKR for real-time data and order execution.
    """
    config = get_config(ctx.obj.get("config_path"))
    setup_logger("trading", config.logging.level, config.logging.file)

    # Override mode
    config.trading.mode = mode

    console.print(Panel(f"Starting trader in [bold]{mode}[/bold] mode"))

    if mode == "live":
        if not click.confirm("Are you sure you want to start LIVE trading?"):
            return

    try:
        from .execution.live_trader import LiveTrader
        from .data.ibkr_client import IBKRClient

        # Connect to IBKR
        ibkr_client = connect_ibkr(config)
        if not ibkr_client:
            console.print("[red]Cannot trade without IBKR connection.[/red]")
            return

        trader = LiveTrader(config, ibkr_client=ibkr_client)

        console.print("[yellow]Press Ctrl+C to stop trading[/yellow]")
        console.print()

        trader.start()

        # Keep running until interrupted
        try:
            while True:
                import time
                time.sleep(10)

                # Print status periodically
                status = trader.get_status()
                console.print(
                    f"Positions: {status['open_positions']} | "
                    f"Daily P&L: ${status['daily_pnl']:.2f} | "
                    f"Trades: {status['daily_trades']}"
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping trader...[/yellow]")
            trader.stop()

    except Exception as e:
        console.print(f"[red]Trading error: {e}[/red]")
        raise


@cli.command()
@click.pass_context
def connect(ctx: click.Context) -> None:
    """Test connection to IBKR."""
    config = get_config(ctx.obj.get("config_path"))

    ibkr_client = connect_ibkr(config)
    if ibkr_client:
        try:
            # Get account info
            summary = ibkr_client.get_account_summary()
            positions = ibkr_client.get_positions()

            table = Table(title="IBKR Connection Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Connected", "Yes")
            table.add_row("Host", config.ibkr.host)
            table.add_row("Port", str(config.ibkr.port))
            table.add_row("Net Liquidation", f"${summary.get('NetLiquidation', 0):,.2f}")
            table.add_row("Available Funds", f"${summary.get('AvailableFunds', 0):,.2f}")
            table.add_row("Open Positions", str(len(positions)))

            console.print(table)

        finally:
            ibkr_client.disconnect()
            console.print("[dim]Disconnected from IBKR[/dim]")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """View system status."""
    config = get_config(ctx.obj.get("config_path"))
    database = Database(config.database_path)

    # Pattern statistics
    all_patterns = database.get_patterns()
    pending = len([p for p in all_patterns if p["status"] == "pending"])
    backtested = len([p for p in all_patterns if p["status"] == "backtested"])
    deployed = len([p for p in all_patterns if p["status"] == "deployed"])
    failed = len([p for p in all_patterns if p["status"] == "failed"])

    table = Table(title="System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Patterns", str(len(all_patterns)))
    table.add_row("Pending", str(pending))
    table.add_row("Backtested (Passed)", str(backtested))
    table.add_row("Deployed", str(deployed))
    table.add_row("Failed", str(failed))
    table.add_row("Trading Mode", config.trading.mode)
    table.add_row("Symbols", ", ".join(config.trading.symbols))
    table.add_row("IBKR Host", f"{config.ibkr.host}:{config.ibkr.port}")
    table.add_row("Data Source", "IBKR (consistent)")

    console.print(table)


@cli.command()
@click.pass_context
def patterns(ctx: click.Context) -> None:
    """List all patterns."""
    config = get_config(ctx.obj.get("config_path"))
    database = Database(config.database_path)

    all_patterns = database.get_patterns()

    if not all_patterns:
        console.print("[yellow]No patterns found. Use 'generate' command to create patterns.[/yellow]")
        return

    table = Table(title="Trading Patterns")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Created")

    for p in all_patterns:
        status_color = {
            "pending": "yellow",
            "backtested": "green",
            "deployed": "blue",
            "failed": "red",
        }.get(p["status"], "white")

        table.add_row(
            p["id"][:8],
            p["name"][:30],
            p.get("pattern_type", "N/A"),
            p.get("symbol", "N/A"),
            f"[{status_color}]{p['status']}[/{status_color}]",
            str(p.get("created_at", "N/A"))[:10],
        )

    console.print(table)


@cli.command()
@click.pass_context
def performance(ctx: click.Context) -> None:
    """View trading performance."""
    config = get_config(ctx.obj.get("config_path"))
    database = Database(config.database_path)

    trades = database.get_trades(mode=config.trading.mode, limit=1000)

    if not trades:
        console.print("[yellow]No trades found.[/yellow]")
        return

    # Calculate metrics
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]

    table = Table(title="Trading Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(len(trades)))
    table.add_row("Winning Trades", str(len(winning)))
    table.add_row("Losing Trades", str(len(losing)))
    table.add_row("Win Rate", f"{len(winning)/len(trades):.1%}" if trades else "N/A")
    table.add_row("Total P&L", f"${sum(pnls):.2f}")
    table.add_row("Average Win", f"${sum(winning)/len(winning):.2f}" if winning else "N/A")
    table.add_row("Average Loss", f"${sum(losing)/len(losing):.2f}" if losing else "N/A")
    table.add_row("Largest Win", f"${max(pnls):.2f}" if pnls else "N/A")
    table.add_row("Largest Loss", f"${min(pnls):.2f}" if pnls else "N/A")

    console.print(table)


@cli.command()
@click.pass_context
def positions(ctx: click.Context) -> None:
    """View open positions."""
    console.print("[yellow]No live trader running. Start with 'trade' command.[/yellow]")


def main() -> None:
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
