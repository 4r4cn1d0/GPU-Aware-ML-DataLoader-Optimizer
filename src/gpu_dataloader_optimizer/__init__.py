"""
GPU-Aware ML DataLoader Optimizer

A profiler that learns optimal data loading patterns across batch sizes, 
formats (TFRecord, CSV, Parquet), and GPU memory limits.
"""

__version__ = "0.1.0"
__author__ = "GPU DataLoader Optimizer Team"

from .core.profiler import DataLoaderProfiler
from .core.optimizer import DataLoaderOptimizer
from .loaders import TFRecordLoader, CSVLoader, ParquetLoader
from .monitoring import GPUMonitor, SystemMonitor
from .benchmarks import ModelBenchmark
from .utils import config_utils, memory_utils

__all__ = [
    "DataLoaderProfiler",
    "DataLoaderOptimizer", 
    "TFRecordLoader",
    "CSVLoader",
    "ParquetLoader",
    "GPUMonitor",
    "SystemMonitor",
    "ModelBenchmark",
    "config_utils",
    "memory_utils",
] 