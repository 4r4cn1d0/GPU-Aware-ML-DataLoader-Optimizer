"""
Base model benchmarking framework for evaluating data loader performance.
"""

import time
import gc
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..monitoring.gpu_monitor import GPUMonitor
from ..monitoring.system_monitor import SystemMonitor
from ..utils.memory_utils import MemoryProfiler


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a model benchmark run."""
    
    # Model and configuration
    model_name: str
    batch_size: int
    data_format: str
    loader_config: Dict[str, Any]
    
    # Performance metrics
    avg_batch_time_sec: float
    std_batch_time_sec: float
    throughput_samples_per_sec: float
    throughput_batches_per_sec: float
    
    # Model-specific metrics
    forward_time_sec: float
    backward_time_sec: float
    optimizer_time_sec: float
    data_loading_time_sec: float
    
    # Resource usage
    peak_gpu_memory_mb: float
    avg_gpu_memory_mb: float
    peak_cpu_memory_mb: float
    avg_cpu_memory_mb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    
    # Quality metrics
    batches_processed: int
    samples_processed: int
    errors_encountered: int
    
    # Model performance (if applicable)
    loss_values: List[float] = field(default_factory=list)
    accuracy_values: List[float] = field(default_factory=list)
    
    # Timing breakdown
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    device: str = "cuda"
    model_parameters: int = 0
    model_flops: Optional[int] = None


class ModelBenchmark(ABC):
    """
    Base class for benchmarking ML models with different data loading configurations.
    
    Features:
    - Multi-model support (Vision, NLP, etc.)
    - Comprehensive performance metrics
    - GPU and system resource monitoring
    - Memory profiling
    - Training and inference benchmarks
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        mixed_precision: bool = True,
        compile_model: bool = False
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.compile_model = compile_model
        
        # Model and components
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # Monitoring
        self.gpu_monitor = GPUMonitor()
        self.system_monitor = SystemMonitor()
        self.memory_profiler = MemoryProfiler()
        
        # Initialize model
        self._setup_model()
        
        logger.info(f"ModelBenchmark initialized: {model_name} on {self.device}")
    
    @abstractmethod
    def _setup_model(self):
        """Setup the model, criterion, and optimizer. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_dummy_batch(self, batch_size: int) -> Any:
        """Create a dummy batch for benchmarking. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _process_batch(self, batch: Any, training: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process a batch and return loss and metrics. Must be implemented by subclasses."""
        pass
    
    def benchmark_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: int = 50,
        training: bool = True,
        warmup_batches: int = 5
    ) -> BenchmarkResult:
        """
        Benchmark a dataloader with the model.
        
        Args:
            dataloader: DataLoader to benchmark
            num_batches: Number of batches to process
            training: Whether to run in training mode
            warmup_batches: Number of warmup batches
        
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        logger.info(f"Benchmarking {self.model_name} with {num_batches} batches (training={training})")
        
        # Set model mode
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        # Initialize monitoring
        self.gpu_monitor.start_monitoring()
        self.system_monitor.start_monitoring()
        self.memory_profiler.start_profiling(f"benchmark_{self.model_name}")
        
        # Timing and metrics
        batch_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        data_loading_times = []
        
        loss_values = []
        accuracy_values = []
        timing_breakdown = {}
        
        total_samples = 0
        errors = 0
        
        try:
            # Warmup phase
            logger.info(f"Warming up with {warmup_batches} batches...")
            warmup_iterator = iter(dataloader)
            
            for i in range(warmup_batches):
                try:
                    batch = next(warmup_iterator)
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        loss, metrics = self._process_batch(batch, training=training)
                    
                    if training:
                        self._backward_pass(loss)
                    
                    # Clean up
                    del batch, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except StopIteration:
                    break
                except Exception as e:
                    logger.warning(f"Warmup batch {i} failed: {e}")
            
            # Main benchmark phase
            logger.info(f"Running benchmark with {num_batches} batches...")
            benchmark_iterator = iter(dataloader)
            
            for batch_idx in range(num_batches):
                try:
                    # Time data loading
                    data_start = time.perf_counter()
                    batch = next(benchmark_iterator)
                    data_end = time.perf_counter()
                    data_loading_time = data_end - data_start
                    
                    # Move to device and get batch size
                    batch = self._move_to_device(batch)
                    batch_size = self._get_batch_size(batch)
                    total_samples += batch_size
                    
                    # Time full batch processing
                    batch_start = time.perf_counter()
                    
                    # Forward pass
                    forward_start = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                        loss, metrics = self._process_batch(batch, training=training)
                    forward_end = time.perf_counter()
                    forward_time = forward_end - forward_start
                    
                    # Backward pass (if training)
                    backward_time = 0.0
                    optimizer_time = 0.0
                    
                    if training:
                        backward_start = time.perf_counter()
                        self._backward_pass(loss)
                        backward_end = time.perf_counter()
                        backward_time = backward_end - backward_start
                        
                        # Optimizer step
                        optimizer_start = time.perf_counter()
                        self._optimizer_step()
                        optimizer_end = time.perf_counter()
                        optimizer_time = optimizer_end - optimizer_start
                    
                    batch_end = time.perf_counter()
                    batch_time = batch_end - batch_start
                    
                    # Record timings
                    batch_times.append(batch_time)
                    forward_times.append(forward_time)
                    backward_times.append(backward_time)
                    optimizer_times.append(optimizer_time)
                    data_loading_times.append(data_loading_time)
                    
                    # Record metrics
                    loss_values.append(loss.item() if torch.is_tensor(loss) else loss)
                    if 'accuracy' in metrics:
                        accuracy_values.append(metrics['accuracy'])
                    
                    # Clean up
                    del batch, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Progress logging
                    if (batch_idx + 1) % 10 == 0:
                        logger.debug(f"Processed {batch_idx + 1}/{num_batches} batches")
                
                except StopIteration:
                    logger.warning(f"DataLoader exhausted at batch {batch_idx}")
                    break
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    errors += 1
                    continue
        
        finally:
            # Stop monitoring
            gpu_stats = self.gpu_monitor.stop_monitoring()
            system_stats = self.system_monitor.stop_monitoring()
            memory_stats = self.memory_profiler.stop_profiling()
        
        # Calculate final metrics
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            std_batch_time = np.std(batch_times)
            throughput_samples = total_samples / sum(batch_times)
            throughput_batches = len(batch_times) / sum(batch_times)
        else:
            avg_batch_time = std_batch_time = throughput_samples = throughput_batches = 0.0
        
        # Create result
        result = BenchmarkResult(
            model_name=self.model_name,
            batch_size=getattr(dataloader, 'batch_size', 0),
            data_format=getattr(dataloader.dataset, 'data_format', 'unknown'),
            loader_config=self._extract_loader_config(dataloader),
            avg_batch_time_sec=avg_batch_time,
            std_batch_time_sec=std_batch_time,
            throughput_samples_per_sec=throughput_samples,
            throughput_batches_per_sec=throughput_batches,
            forward_time_sec=np.mean(forward_times) if forward_times else 0.0,
            backward_time_sec=np.mean(backward_times) if backward_times else 0.0,
            optimizer_time_sec=np.mean(optimizer_times) if optimizer_times else 0.0,
            data_loading_time_sec=np.mean(data_loading_times) if data_loading_times else 0.0,
            peak_gpu_memory_mb=gpu_stats.get('peak_memory_mb', 0.0),
            avg_gpu_memory_mb=gpu_stats.get('avg_memory_mb', 0.0),
            peak_cpu_memory_mb=system_stats.get('peak_memory_mb', 0.0),
            avg_cpu_memory_mb=system_stats.get('avg_memory_mb', 0.0),
            gpu_utilization_percent=gpu_stats.get('avg_utilization', 0.0),
            cpu_utilization_percent=system_stats.get('avg_cpu_percent', 0.0),
            batches_processed=len(batch_times),
            samples_processed=total_samples,
            errors_encountered=errors,
            loss_values=loss_values,
            accuracy_values=accuracy_values,
            timing_breakdown={
                'forward_percent': (np.mean(forward_times) / avg_batch_time) * 100 if avg_batch_time > 0 else 0,
                'backward_percent': (np.mean(backward_times) / avg_batch_time) * 100 if avg_batch_time > 0 else 0,
                'optimizer_percent': (np.mean(optimizer_times) / avg_batch_time) * 100 if avg_batch_time > 0 else 0,
                'data_loading_percent': (np.mean(data_loading_times) / avg_batch_time) * 100 if avg_batch_time > 0 else 0,
            },
            device=str(self.device),
            model_parameters=self._count_parameters(),
            model_flops=self._estimate_flops() if hasattr(self, '_estimate_flops') else None
        )
        
        logger.info(f"Benchmark complete: {throughput_samples:.1f} samples/sec, "
                   f"{result.peak_gpu_memory_mb:.0f}MB peak GPU memory")
        
        return result
    
    def benchmark_synthetic_data(
        self,
        batch_sizes: List[int],
        num_batches: int = 50,
        training: bool = True
    ) -> List[BenchmarkResult]:
        """Benchmark with synthetic data across different batch sizes."""
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking synthetic data with batch_size={batch_size}")
            
            # Create synthetic dataloader
            synthetic_dataloader = self._create_synthetic_dataloader(batch_size, num_batches)
            
            # Run benchmark
            result = self.benchmark_dataloader(
                synthetic_dataloader,
                num_batches=num_batches,
                training=training
            )
            
            results.append(result)
        
        return results
    
    def compare_dataloaders(
        self,
        dataloaders: Dict[str, DataLoader],
        num_batches: int = 50,
        training: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple dataloaders."""
        results = {}
        
        for name, dataloader in dataloaders.items():
            logger.info(f"Benchmarking dataloader: {name}")
            
            try:
                result = self.benchmark_dataloader(
                    dataloader,
                    num_batches=num_batches,
                    training=training
                )
                results[name] = result
                
            except Exception as e:
                logger.error(f"Failed to benchmark {name}: {e}")
        
        return results
    
    def _move_to_device(self, batch: Any) -> Any:
        """Move batch to the appropriate device."""
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        else:
            return batch
    
    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch."""
        if torch.is_tensor(batch):
            return batch.shape[0]
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            return self._get_batch_size(batch[0])
        elif isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value):
                    return value.shape[0]
        return 1
    
    def _backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with optional mixed precision."""
        if self.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Perform optimizer step with optional mixed precision."""
        if self.mixed_precision and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _count_parameters(self) -> int:
        """Count total number of model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _extract_loader_config(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Extract configuration from dataloader."""
        config = {
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'drop_last': dataloader.drop_last,
        }
        
        # Add dataset-specific info if available
        if hasattr(dataloader.dataset, '__class__'):
            config['dataset_type'] = dataloader.dataset.__class__.__name__
        
        if hasattr(dataloader.dataset, 'data_format'):
            config['data_format'] = dataloader.dataset.data_format
        
        return config
    
    def _create_synthetic_dataloader(self, batch_size: int, num_batches: int) -> DataLoader:
        """Create a synthetic dataloader for benchmarking."""
        from torch.utils.data import TensorDataset
        
        # Create synthetic data
        total_samples = batch_size * num_batches
        dummy_batch = self._create_dummy_batch(batch_size)
        
        # Create dataset from dummy batch
        if torch.is_tensor(dummy_batch):
            # Single tensor
            data = dummy_batch.repeat(num_batches, *[1] * (dummy_batch.dim() - 1))
            targets = torch.randint(0, 10, (total_samples,))  # Random targets
            dataset = TensorDataset(data, targets)
        
        elif isinstance(dummy_batch, (list, tuple)):
            # Multiple tensors
            tensors = []
            for item in dummy_batch:
                if torch.is_tensor(item):
                    repeated = item.repeat(num_batches, *[1] * (item.dim() - 1))
                    tensors.append(repeated)
            dataset = TensorDataset(*tensors)
        
        else:
            raise ValueError(f"Unsupported dummy batch type: {type(dummy_batch)}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # No multiprocessing for synthetic data
            pin_memory=torch.cuda.is_available()
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        if self.model is None:
            return {}
        
        info = {
            'model_name': self.model_name,
            'model_class': self.model.__class__.__name__,
            'total_parameters': self._count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'compiled': self.compile_model
        }
        
        # Add memory usage
        if torch.cuda.is_available():
            # Rough estimate of model memory
            param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
            info['estimated_memory_mb'] = param_memory / (1024**2)
        
        return info
    
    def profile_model_forward(self, batch_size: int = 32, num_iterations: int = 100) -> Dict[str, Any]:
        """Profile just the model forward pass."""
        dummy_batch = self._create_dummy_batch(batch_size)
        dummy_batch = self._move_to_device(dummy_batch)
        
        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Profile
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    output = self.model(dummy_batch)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                del output
        
        return {
            'avg_forward_time_sec': np.mean(times),
            'std_forward_time_sec': np.std(times),
            'min_forward_time_sec': np.min(times),
            'max_forward_time_sec': np.max(times),
            'throughput_samples_per_sec': (batch_size * num_iterations) / sum(times),
            'batch_size': batch_size,
            'iterations': num_iterations
        } 