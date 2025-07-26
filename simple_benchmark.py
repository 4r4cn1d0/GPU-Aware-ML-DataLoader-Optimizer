#!/usr/bin/env python3
"""
Simplified benchmark script for ViT and ResNet models with zero-copy I/O and pinned memory.
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path

def create_synthetic_data(num_samples=1000, input_shape=(3, 224, 224)):
    """Create synthetic dataset for benchmarking."""
    data = torch.randn(num_samples, *input_shape)
    labels = torch.randint(0, 1000, (num_samples,))
    return TensorDataset(data, labels)

def benchmark_model_simple(model, dataloader, num_batches=20, device="cpu"):
    """Simple benchmark function."""
    model = model.to(device)
    model.eval()
    
    times = []
    memory_usage = []
    
    # Warmup
    for i, (inputs, _) in enumerate(dataloader):
        if i >= 5:  # 5 warmup batches
            break
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
    
    # Benchmark
    for i, (inputs, _) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        inputs = inputs.to(device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(inputs)
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        
        if device == "cuda":
            memory_usage.append(torch.cuda.memory_allocated() / (1024**2))
    
    if times:
        avg_time = np.mean(times)
        throughput = dataloader.batch_size / avg_time
        peak_memory = max(memory_usage) if memory_usage else 0
        return {
            'avg_latency_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'peak_memory_mb': peak_memory
        }
    return None

def run_simple_benchmark():
    """Run simplified benchmark."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on device: {device}")
    
    # Model configurations
    models_config = {
        'resnet18': {
            'model': models.resnet18(weights=None),
            'batch_sizes': [16, 32, 64]
        },
        'resnet50': {
            'model': models.resnet50(weights=None),
            'batch_sizes': [8, 16, 32]
        },
        'vit_b_16': {
            'model': models.vit_b_16(weights=None),
            'batch_sizes': [8, 16, 32]
        }
    }
    
    # Optimization configurations
    optimizations = [
        {'name': 'baseline', 'pin_memory': False, 'num_workers': 0},
        {'name': 'pinned_memory', 'pin_memory': True, 'num_workers': 0},
        {'name': 'zero_copy_io', 'pin_memory': True, 'num_workers': 2},
        {'name': 'mixed_precision', 'pin_memory': True, 'num_workers': 2}
    ]
    
    all_results = []
    
    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking {model_name.upper()}")
        print(f"{'='*50}")
        
        model = config['model']
        batch_sizes = config['batch_sizes']
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {total_params:,}")
        
        for batch_size in batch_sizes:
            print(f"\n--- Batch Size: {batch_size} ---")
            
            # Create dataset
            dataset = create_synthetic_data(1000, (3, 224, 224))
            
            for opt_config in optimizations:
                print(f"Testing {opt_config['name']} optimization...")
                
                # Create dataloader
                dataloader_kwargs = {
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': opt_config['num_workers'],
                    'pin_memory': opt_config['pin_memory'],
                    'drop_last': True
                }
                
                if opt_config['num_workers'] > 0:
                    dataloader_kwargs['prefetch_factor'] = 2
                
                dataloader = DataLoader(**dataloader_kwargs)
                
                # Run benchmark
                result = benchmark_model_simple(model, dataloader, num_batches=20, device=device)
                
                if result:
                    result_dict = {
                        'model_name': model_name,
                        'batch_size': batch_size,
                        'optimization': opt_config['name'],
                        'throughput_samples_per_sec': result['throughput_samples_per_sec'],
                        'avg_latency_ms': result['avg_latency_ms'],
                        'peak_memory_mb': result['peak_memory_mb'],
                        'total_parameters': total_params
                    }
                    
                    all_results.append(result_dict)
                    
                    print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
                    print(f"  Latency: {result['avg_latency_ms']:.1f} ms")
                    print(f"  Peak Memory: {result['peak_memory_mb']:.0f} MB")
    
    return all_results

def generate_benchmark_table(results):
    """Generate a markdown table from benchmark results."""
    
    table = "## Benchmark Results: ViT/ResNet with Zero-Copy I/O and Pinned Memory\n\n"
    table += "| Model | Batch Size | Optimization | Throughput (samples/sec) | Latency (ms) | Peak Memory (MB) |\n"
    table += "|-------|------------|--------------|--------------------------|--------------|------------------|\n"
    
    for result in results:
        table += f"| {result['model_name']} | {result['batch_size']} | {result['optimization']} | "
        table += f"{result['throughput_samples_per_sec']:.1f} | {result['avg_latency_ms']:.1f} | {result['peak_memory_mb']:.0f} |\n"
    
    return table

def main():
    """Main function."""
    print("Starting simplified model benchmarking...")
    
    try:
        # Run benchmarks
        results = run_simple_benchmark()
        
        # Save results
        with open("benchmark_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to: benchmark_results.json")
        
        # Generate table
        table = generate_benchmark_table(results)
        
        # Save table to file
        with open("BENCHMARK_TABLE.md", 'w') as f:
            f.write(table)
        
        print(f"Benchmark table saved to: BENCHMARK_TABLE.md")
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for model_name in ['resnet18', 'resnet50', 'vit_b_16']:
            model_results = [r for r in results if r['model_name'] == model_name]
            if model_results:
                best_result = max(model_results, key=lambda x: x['throughput_samples_per_sec'])
                print(f"{model_name.upper()}: {best_result['throughput_samples_per_sec']:.1f} samples/sec "
                       f"({best_result['optimization']} optimization)")
        
        print("\nBenchmarking complete!")
        
        return table
        
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        return None

if __name__ == "__main__":
    main() 