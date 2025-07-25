"""
Core profiler that measures data loading performance across different configurations.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import psutil
from rich.console import Console
from rich.progress import Progress, TaskID

from ..monitoring.gpu_monitor import GPUMonitor
from ..monitoring.system_monitor import SystemMonitor
from ..utils.memory_utils import MemoryProfiler
from ..utils.config_utils import ProfilerConfig


logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ProfilingResult:
    """Results from a single profiling run."""
    
    # Configuration
    batch_size: int
    data_format: str
    loader_config: Dict[str, Any]
    
    # Performance metrics
    avg_load_time: float
    std_load_time: float
    throughput_samples_per_sec: float
    throughput_mb_per_sec: float
    
    # Resource usage
    peak_gpu_memory_mb: float
    avg_gpu_memory_mb: float
    peak_cpu_memory_mb: float
    avg_cpu_memory_mb: float
    cpu_utilization: float
    gpu_utilization: float
    
    # System metrics
    io_read_mb: float
    io_write_mb: float
    cache_hit_rate: float
    
    # Timing breakdown
    data_fetch_time: float
    preprocessing_time: float
    gpu_transfer_time: float
    
    # Quality metrics
    samples_processed: int
    errors_encountered: int
    stability_score: float  # Consistency across runs
    
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ProfilingSession:
    """A complete profiling session with multiple configurations."""
    
    session_id: str
    results: List[ProfilingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add_result(self, result: ProfilingResult):
        """Add a profiling result to this session."""
        self.results.append(result)
    
    def finalize(self):
        """Mark the session as complete."""
        self.end_time = time.time()
    
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


class DataLoaderProfiler:
    """
    Core profiler that measures data loading performance across different configurations.
    
    Features:
    - Multi-threaded profiling for parallel evaluation
    - GPU and system resource monitoring
    - Memory profiling with peak/average tracking
    - I/O pattern analysis
    - Statistical analysis of performance variations
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.gpu_monitor = GPUMonitor()
        self.system_monitor = SystemMonitor()
        self.memory_profiler = MemoryProfiler()
        
        self.current_session: Optional[ProfilingSession] = None
        self.sessions: List[ProfilingSession] = []
        
        # State tracking
        self._is_profiling = False
        self._profile_lock = threading.Lock()
        
        logger.info("DataLoaderProfiler initialized")
    
    def start_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> ProfilingSession:
        """Start a new profiling session."""
        with self._profile_lock:
            if self.current_session and not self.current_session.end_time:
                logger.warning(f"Finalizing previous session: {self.current_session.session_id}")
                self.current_session.finalize()
            
            self.current_session = ProfilingSession(
                session_id=session_id,
                metadata=metadata or {}
            )
            self.sessions.append(self.current_session)
            
            logger.info(f"Started profiling session: {session_id}")
            return self.current_session
    
    def profile_configuration(
        self,
        dataloader_factory: Callable,
        batch_size: int,
        data_format: str,
        loader_config: Dict[str, Any],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> ProfilingResult:
        """
        Profile a specific dataloader configuration.
        
        Args:
            dataloader_factory: Function that creates the dataloader
            batch_size: Batch size to test
            data_format: Data format (tfrecord, csv, parquet)
            loader_config: Configuration for the loader
            num_iterations: Number of iterations to profile
            warmup_iterations: Number of warmup iterations
        
        Returns:
            ProfilingResult with comprehensive metrics
        """
        if not self.current_session:
            raise RuntimeError("No active profiling session. Call start_session() first.")
        
        logger.info(f"Profiling configuration: batch_size={batch_size}, format={data_format}")
        
        # Create dataloader
        dataloader = dataloader_factory(batch_size=batch_size, **loader_config)
        
        # Initialize monitoring
        self.gpu_monitor.start_monitoring()
        self.system_monitor.start_monitoring()
        
        # Timing and performance tracking
        load_times = []
        fetch_times = []
        preprocess_times = []
        transfer_times = []
        
        total_samples = 0
        total_bytes = 0
        errors = 0
        
        try:
            # Warmup phase
            console.print(f"[yellow]Warming up ({warmup_iterations} iterations)...")
            for _ in range(warmup_iterations):
                try:
                    batch = next(iter(dataloader))
                    del batch  # Free memory immediately
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
            
            # Main profiling phase
            console.print(f"[green]Profiling ({num_iterations} iterations)...")
            
            with Progress() as progress:
                task = progress.add_task("Profiling...", total=num_iterations)
                
                for i in range(num_iterations):
                    try:
                        # Time the full loading process
                        start_time = time.perf_counter()
                        
                        # Detailed timing breakdown
                        fetch_start = time.perf_counter()
                        batch = next(iter(dataloader))
                        fetch_end = time.perf_counter()
                        
                        # Simulate preprocessing (if any)
                        preprocess_start = time.perf_counter()
                        # This would be actual preprocessing in real usage
                        preprocess_end = time.perf_counter()
                        
                        # Simulate GPU transfer
                        transfer_start = time.perf_counter()
                        if hasattr(batch, 'cuda') and callable(batch.cuda):
                            batch = batch.cuda(non_blocking=True)
                        elif isinstance(batch, (list, tuple)):
                            # Handle batches that are lists/tuples
                            pass
                        transfer_end = time.perf_counter()
                        
                        end_time = time.perf_counter()
                        
                        # Record timings
                        total_time = end_time - start_time
                        load_times.append(total_time)
                        fetch_times.append(fetch_end - fetch_start)
                        preprocess_times.append(preprocess_end - preprocess_start)
                        transfer_times.append(transfer_end - transfer_start)
                        
                        # Calculate batch metrics
                        batch_samples = self._get_batch_size(batch)
                        batch_bytes = self._estimate_batch_bytes(batch)
                        
                        total_samples += batch_samples
                        total_bytes += batch_bytes
                        
                        # Clean up
                        del batch
                        
                    except Exception as e:
                        logger.error(f"Profiling iteration {i} failed: {e}")
                        errors += 1
                    
                    progress.update(task, advance=1)
        
        finally:
            # Stop monitoring
            gpu_stats = self.gpu_monitor.stop_monitoring()
            system_stats = self.system_monitor.stop_monitoring()
        
        # Calculate performance metrics
        avg_load_time = np.mean(load_times) if load_times else 0.0
        std_load_time = np.std(load_times) if load_times else 0.0
        
        total_time = sum(load_times)
        throughput_samples = total_samples / total_time if total_time > 0 else 0.0
        throughput_mb = (total_bytes / (1024**2)) / total_time if total_time > 0 else 0.0
        
        # Stability score (inverse of coefficient of variation)
        cv = std_load_time / avg_load_time if avg_load_time > 0 else float('inf')
        stability_score = 1.0 / (1.0 + cv)
        
        # Create result
        result = ProfilingResult(
            batch_size=batch_size,
            data_format=data_format,
            loader_config=loader_config,
            avg_load_time=avg_load_time,
            std_load_time=std_load_time,
            throughput_samples_per_sec=throughput_samples,
            throughput_mb_per_sec=throughput_mb,
            peak_gpu_memory_mb=gpu_stats.get('peak_memory_mb', 0.0),
            avg_gpu_memory_mb=gpu_stats.get('avg_memory_mb', 0.0),
            peak_cpu_memory_mb=system_stats.get('peak_memory_mb', 0.0),
            avg_cpu_memory_mb=system_stats.get('avg_memory_mb', 0.0),
            cpu_utilization=system_stats.get('avg_cpu_percent', 0.0),
            gpu_utilization=gpu_stats.get('avg_utilization', 0.0),
            io_read_mb=system_stats.get('io_read_mb', 0.0),
            io_write_mb=system_stats.get('io_write_mb', 0.0),
            cache_hit_rate=system_stats.get('cache_hit_rate', 0.0),
            data_fetch_time=np.mean(fetch_times) if fetch_times else 0.0,
            preprocessing_time=np.mean(preprocess_times) if preprocess_times else 0.0,
            gpu_transfer_time=np.mean(transfer_times) if transfer_times else 0.0,
            samples_processed=total_samples,
            errors_encountered=errors,
            stability_score=stability_score
        )
        
        self.current_session.add_result(result)
        
        console.print(f"[green]✓ Configuration profiled: {throughput_samples:.1f} samples/sec, "
                     f"{throughput_mb:.1f} MB/sec")
        
        return result
    
    def profile_batch_sizes(
        self,
        dataloader_factory: Callable,
        batch_sizes: List[int],
        data_format: str,
        loader_config: Dict[str, Any],
        parallel: bool = True
    ) -> List[ProfilingResult]:
        """Profile multiple batch sizes."""
        results = []
        
        if parallel and len(batch_sizes) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(batch_sizes))) as executor:
                futures = []
                for batch_size in batch_sizes:
                    future = executor.submit(
                        self.profile_configuration,
                        dataloader_factory, batch_size, data_format, loader_config
                    )
                    futures.append(future)
                
                for future in futures:
                    results.append(future.result())
        else:
            for batch_size in batch_sizes:
                result = self.profile_configuration(
                    dataloader_factory, batch_size, data_format, loader_config
                )
                results.append(result)
        
        return results
    
    def finalize_session(self) -> Optional[ProfilingSession]:
        """Finalize the current profiling session."""
        with self._profile_lock:
            if self.current_session:
                self.current_session.finalize()
                session = self.current_session
                self.current_session = None
                
                logger.info(f"Finalized session: {session.session_id}, "
                           f"duration: {session.duration():.1f}s, "
                           f"results: {len(session.results)}")
                
                return session
            return None
    
    def save_session(self, session: ProfilingSession, filepath: Path):
        """Save a profiling session to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(session, f)
        
        logger.info(f"Saved session to: {filepath}")
    
    def load_session(self, filepath: Path) -> ProfilingSession:
        """Load a profiling session from disk."""
        with open(filepath, 'rb') as f:
            session = pickle.load(f)
        
        logger.info(f"Loaded session: {session.session_id}")
        return session
    
    def export_results_json(self, session: ProfilingSession, filepath: Path):
        """Export session results to JSON format."""
        data = {
            'session_id': session.session_id,
            'metadata': session.metadata,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration': session.duration(),
            'results': []
        }
        
        for result in session.results:
            result_dict = {
                'batch_size': result.batch_size,
                'data_format': result.data_format,
                'loader_config': result.loader_config,
                'performance': {
                    'avg_load_time': result.avg_load_time,
                    'std_load_time': result.std_load_time,
                    'throughput_samples_per_sec': result.throughput_samples_per_sec,
                    'throughput_mb_per_sec': result.throughput_mb_per_sec,
                    'stability_score': result.stability_score
                },
                'resources': {
                    'peak_gpu_memory_mb': result.peak_gpu_memory_mb,
                    'avg_gpu_memory_mb': result.avg_gpu_memory_mb,
                    'peak_cpu_memory_mb': result.peak_cpu_memory_mb,
                    'avg_cpu_memory_mb': result.avg_cpu_memory_mb,
                    'cpu_utilization': result.cpu_utilization,
                    'gpu_utilization': result.gpu_utilization
                },
                'io': {
                    'io_read_mb': result.io_read_mb,
                    'io_write_mb': result.io_write_mb,
                    'cache_hit_rate': result.cache_hit_rate
                },
                'timing_breakdown': {
                    'data_fetch_time': result.data_fetch_time,
                    'preprocessing_time': result.preprocessing_time,
                    'gpu_transfer_time': result.gpu_transfer_time
                },
                'quality': {
                    'samples_processed': result.samples_processed,
                    'errors_encountered': result.errors_encountered
                },
                'timestamp': result.timestamp
            }
            data['results'].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported results to: {filepath}")
    
    def _get_batch_size(self, batch) -> int:
        """Extract batch size from a batch object."""
        if hasattr(batch, 'shape'):
            return batch.shape[0]
        elif hasattr(batch, '__len__'):
            return len(batch)
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            return self._get_batch_size(batch[0])
        else:
            return 1
    
    def _estimate_batch_bytes(self, batch) -> int:
        """Estimate memory usage of a batch in bytes."""
        try:
            if hasattr(batch, 'nbytes'):
                return batch.nbytes
            elif hasattr(batch, 'element_size') and hasattr(batch, 'numel'):
                return batch.element_size() * batch.numel()
            elif isinstance(batch, (list, tuple)):
                return sum(self._estimate_batch_bytes(item) for item in batch)
            else:
                # Rough estimate
                return 1024  # 1KB default
        except Exception:
            return 1024 