"""
Command-line interface for the GPU-Aware ML DataLoader Optimizer.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

from .core.profiler import DataLoaderProfiler
from .core.optimizer import DataLoaderOptimizer
from .loaders import TFRecordLoader, CSVLoader, ParquetLoader
from .utils.config_utils import ConfigManager, SystemConfig, ExperimentConfig
from .utils.memory_utils import get_memory_info, optimize_memory_usage


# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.pass_context
def cli(ctx, config, log_level, output_dir):
    """GPU-Aware ML DataLoader Optimizer - Profile and optimize data loading for ML training."""
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Load configuration
    config_manager = ConfigManager(Path(config) if config else None)
    system_config = config_manager.load_config()
    
    if output_dir:
        system_config.output_dir = Path(output_dir)
        system_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = system_config
    ctx.obj['config_manager'] = config_manager
    
    console.print("[bold green]GPU-Aware ML DataLoader Optimizer[/bold green]")
    console.print(f"Output directory: {system_config.output_dir}")


@cli.command()
@click.argument('data_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--data-format', type=click.Choice(['csv', 'parquet', 'tfrecord']), required=True)
@click.option('--batch-sizes', default='16,32,64,128', help='Comma-separated batch sizes to test')
@click.option('--iterations', default=100, help='Number of iterations per configuration')
@click.option('--target-column', help='Target column name (for supervised learning)')
@click.option('--feature-columns', help='Comma-separated feature column names')
@click.option('--experiment-name', default='dataloader_profile', help='Name for this experiment')
@click.pass_context
def profile(ctx, data_paths, data_format, batch_sizes, iterations, target_column, feature_columns, experiment_name):
    """Profile data loading performance across different configurations."""
    
    config: SystemConfig = ctx.obj['config']
    
    # Parse parameters
    batch_sizes = [int(bs.strip()) for bs in batch_sizes.split(',')]
    feature_cols = [col.strip() for col in feature_columns.split(',')] if feature_columns else None
    
    console.print(f"\n[bold]Profiling Data Loading Performance[/bold]")
    console.print(f"Data paths: {list(data_paths)}")
    console.print(f"Data format: {data_format}")
    console.print(f"Batch sizes: {batch_sizes}")
    console.print(f"Iterations: {iterations}")
    
    # Create profiler
    profiler = DataLoaderProfiler(config.profiler)
    
    # Start profiling session
    session = profiler.start_session(
        experiment_name,
        metadata={
            'data_paths': list(data_paths),
            'data_format': data_format,
            'batch_sizes': batch_sizes,
            'target_column': target_column,
            'feature_columns': feature_cols
        }
    )
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for batch_size in batch_sizes:
                task = progress.add_task(f"Profiling batch_size={batch_size}", total=1)
                
                # Create dataloader factory
                loader_factory = _create_loader_factory(
                    data_paths, data_format, target_column, feature_cols, config.loader
                )
                
                # Profile this configuration
                result = profiler.profile_configuration(
                    loader_factory,
                    batch_size=batch_size,
                    data_format=data_format,
                    loader_config=config.loader.dict(),
                    num_iterations=iterations,
                    warmup_iterations=config.profiler.warmup_iterations
                )
                
                progress.update(task, completed=1)
                
                # Display quick results
                console.print(f"  ✓ Batch size {batch_size}: {result.throughput_samples_per_sec:.1f} samples/sec, "
                            f"{result.peak_gpu_memory_mb:.0f}MB peak GPU memory")
    
    finally:
        # Finalize session
        session = profiler.finalize_session()
        
        if session:
            # Save results
            results_dir = config.output_dir / "profiling_results"
            results_dir.mkdir(exist_ok=True)
            
            session_file = results_dir / f"{experiment_name}_session.pkl"
            profiler.save_session(session, session_file)
            
            json_file = results_dir / f"{experiment_name}_results.json"
            profiler.export_results_json(session, json_file)
            
            console.print(f"\n[green]✓ Profiling complete![/green]")
            console.print(f"Results saved to: {results_dir}")
            
            # Display summary table
            _display_profiling_summary(session.results)


@cli.command()
@click.argument('session_file', type=click.Path(exists=True))
@click.option('--target', type=click.Choice(['speed', 'memory', 'stability', 'balanced']), default='balanced')
@click.option('--max-memory-mb', type=float, help='Maximum GPU memory constraint (MB)')
@click.option('--min-throughput', type=float, help='Minimum throughput constraint (samples/sec)')
@click.pass_context
def optimize(ctx, session_file, target, max_memory_mb, min_throughput):
    """Optimize dataloader configuration based on profiling results."""
    
    config: SystemConfig = ctx.obj['config']
    
    console.print(f"\n[bold]Optimizing DataLoader Configuration[/bold]")
    console.print(f"Session file: {session_file}")
    console.print(f"Optimization target: {target}")
    
    # Load profiling session
    profiler = DataLoaderProfiler(config.profiler)
    session = profiler.load_session(Path(session_file))
    
    console.print(f"Loaded session: {session.session_id} with {len(session.results)} results")
    
    # Create optimizer
    optimizer = DataLoaderOptimizer(config.optimizer)
    
    # Add profiling data
    optimizer.add_profiling_data(session)
    
    # Train models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training optimization models...", total=1)
        optimizer.train_models()
        progress.update(task, completed=1)
    
    # Generate candidate configurations
    candidates = _generate_candidate_configs(session.results)
    
    # Set up constraints
    constraints = {}
    if max_memory_mb:
        constraints['max_memory_mb'] = max_memory_mb
    if min_throughput:
        constraints['min_throughput'] = min_throughput
    
    # Optimize
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing configurations...", total=1)
        report = optimizer.optimize(candidates, constraints, target)
        progress.update(task, completed=1)
    
    # Display results
    console.print(f"\n[green]✓ Optimization complete![/green]")
    
    if report.recommendations:
        _display_optimization_results(report)
        
        # Save optimization report
        results_dir = config.output_dir / "optimization_results"
        results_dir.mkdir(exist_ok=True)
        
        report_file = results_dir / f"optimization_report_{target}.json"
        with open(report_file, 'w') as f:
            # Convert report to JSON-serializable format
            report_dict = {
                'metadata': report.metadata,
                'constraints': report.constraints,
                'analysis': report.analysis,
                'recommendations': [
                    {
                        'rank': rec.rank,
                        'batch_size': rec.batch_size,
                        'data_format': rec.data_format,
                        'loader_config': rec.loader_config,
                        'predicted_throughput': rec.predicted_throughput,
                        'predicted_memory_usage': rec.predicted_memory_usage,
                        'confidence_score': rec.confidence_score,
                        'overall_score': rec.overall_score,
                        'reasoning': rec.reasoning
                    }
                    for rec in report.recommendations[:10]  # Top 10
                ]
            }
            json.dump(report_dict, f, indent=2)
        
        console.print(f"Optimization report saved to: {report_file}")
    
    else:
        console.print("[red]No valid recommendations found![/red]")


@cli.command()
@click.argument('data_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--data-format', type=click.Choice(['csv', 'parquet', 'tfrecord']), required=True)
@click.option('--models', default='resnet18,bert_base', help='Comma-separated model names')
@click.option('--batch-sizes', default='16,32,64', help='Comma-separated batch sizes')
@click.option('--iterations', default=50, help='Number of benchmark iterations')
@click.pass_context
def benchmark(ctx, data_paths, data_format, models, batch_sizes, iterations):
    """Benchmark ML models with different data loading configurations."""
    
    config: SystemConfig = ctx.obj['config']
    
    model_names = [m.strip() for m in models.split(',')]
    batch_sizes = [int(bs.strip()) for bs in batch_sizes.split(',')]
    
    console.print(f"\n[bold]Benchmarking ML Models[/bold]")
    console.print(f"Models: {model_names}")
    console.print(f"Batch sizes: {batch_sizes}")
    
    # This would integrate with the benchmark framework
    console.print("[yellow]Model benchmarking feature coming soon![/yellow]")
    console.print("Use 'profile' command to analyze data loading performance.")


@cli.command()
@click.pass_context
def system_info(ctx):
    """Display system information and hardware compatibility."""
    
    console.print(f"\n[bold]System Information[/bold]")
    
    # Get memory info
    memory_info = get_memory_info()
    
    # Display system memory
    system_mem = memory_info.get('system', {})
    console.print(f"System Memory: {system_mem.get('used_mb', 0):.0f} / {system_mem.get('total_mb', 0):.0f} MB "
                 f"({system_mem.get('percent', 0):.1f}%)")
    
    # Display process memory
    process_mem = memory_info.get('process', {})
    console.print(f"Process Memory: {process_mem.get('rss_mb', 0):.0f} MB RSS, "
                 f"{process_mem.get('vms_mb', 0):.0f} MB VMS")
    
    # Display GPU info
    gpu_info = memory_info.get('gpu', {})
    if gpu_info:
        console.print("\n[bold]GPU Information:[/bold]")
        for device_id, device_info in gpu_info.items():
            console.print(f"  {device_id}: {device_info.get('name', 'Unknown')}")
            console.print(f"    Memory: {device_info.get('allocated_mb', 0):.0f} / "
                         f"{device_info.get('total_mb', 0):.0f} MB")
    else:
        console.print("\n[yellow]No GPU information available[/yellow]")
    
    # Display Python objects
    python_info = memory_info.get('python', {})
    console.print(f"\nPython Objects: {python_info.get('objects_count', 0):,}")
    
    # Hardware compatibility check
    config: SystemConfig = ctx.obj['config']
    warnings = ConfigManager.validate_hardware_compatibility(config)
    
    if warnings:
        console.print("\n[yellow]Hardware Compatibility Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  ⚠️  {warning}")
    else:
        console.print("\n[green]✓ Hardware compatibility looks good![/green]")


@cli.command()
@click.option('--output-file', '-o', type=click.Path(), default='gpu_dataloader_config.yaml')
@click.pass_context
def create_config(ctx, output_file):
    """Create a configuration template file."""
    
    from .utils.config_utils import create_config_template
    
    output_path = Path(output_file)
    create_config_template(output_path)
    
    console.print(f"[green]✓ Configuration template created: {output_path}[/green]")
    console.print("Edit this file to customize your settings, then use --config option.")


@cli.command()
@click.pass_context
def optimize_memory(ctx):
    """Optimize current memory usage."""
    
    console.print("Optimizing memory usage...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running memory optimization...", total=1)
        result = optimize_memory_usage()
        progress.update(task, completed=1)
    
    console.print(f"[green]✓ Memory optimization complete![/green]")
    console.print(f"  Garbage collected: {result['gc_collected']} objects")
    if result['torch_cache_cleared']:
        console.print("  PyTorch GPU cache cleared")


def _create_loader_factory(data_paths, data_format, target_column, feature_columns, config):
    """Create a dataloader factory function."""
    
    def factory(batch_size, **kwargs):
        common_args = {
            'data_path': list(data_paths),
            'batch_size': batch_size,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory,
            'drop_last': config.drop_last,
            'prefetch_factor': config.prefetch_factor,
            'use_memory_mapping': config.use_memory_mapping,
            'cache_size': config.cache_size,
            'enable_zero_copy': config.enable_zero_copy,
            **kwargs
        }
        
        if data_format == 'csv':
            return CSVLoader(
                target_column=target_column,
                feature_columns=feature_columns,
                chunk_size=config.csv_chunk_size,
                use_dask=config.csv_use_dask,
                **common_args
            )
        
        elif data_format == 'parquet':
            return ParquetLoader(
                target_column=target_column,
                feature_columns=feature_columns,
                use_dask=config.parquet_use_dask,
                **common_args
            )
        
        elif data_format == 'tfrecord':
            # Would need feature description - simplified for CLI
            feature_description = {
                'features': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            return TFRecordLoader(
                feature_description=feature_description,
                compression_type=config.tfrecord_compression,
                **common_args
            )
        
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    return factory


def _display_profiling_summary(results):
    """Display profiling results in a table."""
    
    table = Table(title="Profiling Results Summary")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Throughput (samples/sec)", style="green")
    table.add_column("Memory (MB)", style="yellow")
    table.add_column("Load Time (ms)", style="blue")
    table.add_column("Stability", style="magenta")
    
    for result in results:
        table.add_row(
            str(result.batch_size),
            f"{result.throughput_samples_per_sec:.1f}",
            f"{result.peak_gpu_memory_mb:.0f}",
            f"{result.avg_load_time * 1000:.1f}",
            f"{result.stability_score:.3f}"
        )
    
    console.print(table)


def _display_optimization_results(report):
    """Display optimization results."""
    
    console.print("\n[bold]Optimization Recommendations:[/bold]")
    
    table = Table()
    table.add_column("Rank", style="cyan")
    table.add_column("Batch Size", style="green")
    table.add_column("Format", style="yellow")
    table.add_column("Predicted Throughput", style="blue")
    table.add_column("Predicted Memory", style="red")
    table.add_column("Confidence", style="magenta")
    table.add_column("Score", style="white")
    
    for rec in report.recommendations[:5]:  # Top 5
        table.add_row(
            str(rec.rank),
            str(rec.batch_size),
            rec.data_format,
            f"{rec.predicted_throughput:.1f}",
            f"{rec.predicted_memory_usage:.0f}",
            f"{rec.confidence_score:.3f}",
            f"{rec.overall_score:.3f}"
        )
    
    console.print(table)
    
    # Show top recommendation details
    if report.recommendations:
        top_rec = report.recommendations[0]
        console.print(f"\n[bold green]Top Recommendation:[/bold green]")
        console.print(f"  Batch Size: {top_rec.batch_size}")
        console.print(f"  Data Format: {top_rec.data_format}")
        console.print(f"  Predicted Performance: {top_rec.predicted_throughput:.1f} samples/sec")
        console.print(f"  Predicted Memory: {top_rec.predicted_memory_usage:.0f} MB")
        console.print(f"  Confidence: {top_rec.confidence_score:.1%}")
        
        if top_rec.reasoning:
            console.print("\n[bold]Reasoning:[/bold]")
            for reason in top_rec.reasoning:
                console.print(f"  • {reason}")


def _generate_candidate_configs(results):
    """Generate candidate configurations from profiling results."""
    candidates = []
    
    for result in results:
        candidate = {
            'batch_size': result.batch_size,
            'data_format': result.data_format,
            **result.loader_config
        }
        candidates.append(candidate)
    
    # Add some variations
    batch_sizes = list(set(r.batch_size for r in results))
    data_formats = list(set(r.data_format for r in results))
    
    for bs in [16, 32, 64, 128, 256]:
        if bs not in batch_sizes:
            for fmt in data_formats:
                candidates.append({
                    'batch_size': bs,
                    'data_format': fmt,
                    'num_workers': 4,
                    'pin_memory': True,
                    'prefetch_factor': 2
                })
    
    return candidates


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Unexpected error occurred")
        sys.exit(1)


if __name__ == '__main__':
    main() 