"""
Memory utilities for profiling and optimization.
"""

import os
import gc
import mmap
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

import psutil
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None


logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    timestamp: float = field(default_factory=time.time)
    
    # System memory
    system_total_mb: float = 0.0
    system_used_mb: float = 0.0
    system_available_mb: float = 0.0
    system_percent: float = 0.0
    
    # Process memory
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    process_percent: float = 0.0
    
    # GPU memory (if available)
    gpu_total_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_free_mb: float = 0.0
    gpu_percent: float = 0.0
    
    # Python-specific memory
    python_objects_count: int = 0
    python_gc_collected: int = 0
    
    # Custom tags for context
    tags: Dict[str, Any] = field(default_factory=dict)


class MemoryProfiler:
    """
    Advanced memory profiler with GPU awareness and optimization suggestions.
    
    Features:
    - Real-time memory monitoring
    - Peak memory tracking
    - Memory leak detection
    - GPU memory profiling
    - Optimization recommendations
    """
    
    def __init__(self, gpu_id: int = 0, sample_interval: float = 0.1):
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        
        # Initialize GPU monitoring
        self.gpu_available = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.gpu_available = True
                logger.info(f"GPU memory profiling enabled for GPU {gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
        
        # Process handle
        self.process = psutil.Process()
        
        # Profiling state
        self.is_profiling = False
        self.snapshots: List[MemorySnapshot] = []
        self.peak_snapshot: Optional[MemorySnapshot] = None
        
        # Background monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_lock = threading.Lock()
        
    def start_profiling(self, tag: str = "default"):
        """Start continuous memory profiling."""
        with self.monitor_lock:
            if self.is_profiling:
                logger.warning("Memory profiling already active")
                return
            
            self.is_profiling = True
            self.snapshots.clear()
            self.peak_snapshot = None
            
            # Take initial snapshot
            initial_snapshot = self.take_snapshot(tags={"phase": "start", "tag": tag})
            self.snapshots.append(initial_snapshot)
            
            # Start background monitoring
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info(f"Started memory profiling with tag: {tag}")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return analysis."""
        with self.monitor_lock:
            if not self.is_profiling:
                logger.warning("Memory profiling not active")
                return {}
            
            self.is_profiling = False
            
            # Wait for monitor thread
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            
            # Take final snapshot
            final_snapshot = self.take_snapshot(tags={"phase": "end"})
            self.snapshots.append(final_snapshot)
            
            # Analyze results
            analysis = self.analyze_memory_usage()
            
            logger.info(f"Stopped memory profiling. Peak memory: {analysis.get('peak_system_mb', 0):.1f}MB")
            
            return analysis
    
    def take_snapshot(self, tags: Optional[Dict[str, Any]] = None) -> MemorySnapshot:
        """Take a memory snapshot."""
        snapshot = MemorySnapshot(tags=tags or {})
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            snapshot.system_total_mb = system_memory.total / (1024**2)
            snapshot.system_used_mb = system_memory.used / (1024**2)
            snapshot.system_available_mb = system_memory.available / (1024**2)
            snapshot.system_percent = system_memory.percent
            
            # Process memory
            process_memory = self.process.memory_info()
            snapshot.process_rss_mb = process_memory.rss / (1024**2)
            snapshot.process_vms_mb = process_memory.vms / (1024**2)
            snapshot.process_percent = self.process.memory_percent()
            
            # GPU memory
            if self.gpu_available:
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                snapshot.gpu_total_mb = gpu_memory.total / (1024**2)
                snapshot.gpu_used_mb = gpu_memory.used / (1024**2)
                snapshot.gpu_free_mb = gpu_memory.free / (1024**2)
                snapshot.gpu_percent = (gpu_memory.used / gpu_memory.total) * 100
            
            # Python objects
            snapshot.python_objects_count = len(gc.get_objects())
            snapshot.python_gc_collected = gc.get_count()[0]
            
            # Update peak tracking
            if (self.peak_snapshot is None or 
                snapshot.system_used_mb > self.peak_snapshot.system_used_mb):
                self.peak_snapshot = snapshot
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
        
        return snapshot
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_profiling:
            try:
                snapshot = self.take_snapshot(tags={"phase": "monitoring"})
                self.snapshots.append(snapshot)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.snapshots:
            return {}
        
        # Basic statistics
        system_usage = [s.system_used_mb for s in self.snapshots]
        process_usage = [s.process_rss_mb for s in self.snapshots]
        gpu_usage = [s.gpu_used_mb for s in self.snapshots if s.gpu_used_mb > 0]
        
        analysis = {
            'duration_sec': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            'num_snapshots': len(self.snapshots),
            
            # System memory
            'peak_system_mb': max(system_usage) if system_usage else 0,
            'avg_system_mb': np.mean(system_usage) if system_usage else 0,
            'min_system_mb': min(system_usage) if system_usage else 0,
            'system_growth_mb': system_usage[-1] - system_usage[0] if len(system_usage) >= 2 else 0,
            
            # Process memory
            'peak_process_mb': max(process_usage) if process_usage else 0,
            'avg_process_mb': np.mean(process_usage) if process_usage else 0,
            'min_process_mb': min(process_usage) if process_usage else 0,
            'process_growth_mb': process_usage[-1] - process_usage[0] if len(process_usage) >= 2 else 0,
        }
        
        # GPU memory (if available)
        if gpu_usage:
            analysis.update({
                'peak_gpu_mb': max(gpu_usage),
                'avg_gpu_mb': np.mean(gpu_usage),
                'min_gpu_mb': min(gpu_usage),
                'gpu_growth_mb': gpu_usage[-1] - gpu_usage[0] if len(gpu_usage) >= 2 else 0,
            })
        
        # Memory leak detection
        analysis['potential_leak'] = self._detect_memory_leak()
        
        # Optimization recommendations
        analysis['recommendations'] = self._generate_recommendations()
        
        return analysis
    
    def _detect_memory_leak(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 10:
            return {'detected': False, 'confidence': 0.0}
        
        # Analyze growth trend
        recent_snapshots = self.snapshots[-10:]
        memory_values = [s.process_rss_mb for s in recent_snapshots]
        
        # Simple linear regression to detect growth
        x = np.arange(len(memory_values))
        slope = np.polyfit(x, memory_values, 1)[0]
        
        # Consider it a leak if memory grows consistently > 1MB per measurement
        leak_threshold = 1.0  # MB
        leak_detected = slope > leak_threshold
        
        confidence = min(1.0, abs(slope) / leak_threshold) if leak_detected else 0.0
        
        return {
            'detected': leak_detected,
            'confidence': confidence,
            'growth_rate_mb_per_sample': slope,
            'recommendation': 'Check for unreleased objects or growing caches' if leak_detected else None
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if not self.snapshots:
            return recommendations
        
        peak_system = max(s.system_used_mb for s in self.snapshots)
        peak_process = max(s.process_rss_mb for s in self.snapshots)
        peak_gpu = max((s.gpu_used_mb for s in self.snapshots if s.gpu_used_mb > 0), default=0)
        
        # System memory recommendations
        if peak_system > 0.8 * self.snapshots[0].system_total_mb:
            recommendations.append("High system memory usage detected. Consider reducing batch size.")
        
        # Process memory recommendations
        if peak_process > 1000:  # > 1GB
            recommendations.append("High process memory usage. Consider enabling memory mapping or reducing cache size.")
        
        # GPU memory recommendations
        if peak_gpu > 0:
            gpu_total = max((s.gpu_total_mb for s in self.snapshots if s.gpu_total_mb > 0), default=0)
            if peak_gpu > 0.8 * gpu_total:
                recommendations.append("High GPU memory usage. Consider reducing batch size or using gradient checkpointing.")
        
        # Growth recommendations
        process_growth = (self.snapshots[-1].process_rss_mb - self.snapshots[0].process_rss_mb 
                         if len(self.snapshots) >= 2 else 0)
        if process_growth > 100:  # > 100MB growth
            recommendations.append("Significant memory growth detected. Check for memory leaks or growing caches.")
        
        return recommendations
    
    @contextmanager
    def profile_block(self, tag: str = "block"):
        """Context manager for profiling a code block."""
        self.start_profiling(tag)
        try:
            yield self
        finally:
            analysis = self.stop_profiling()
            logger.info(f"Memory profile for '{tag}': peak={analysis.get('peak_process_mb', 0):.1f}MB")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Profile a function call and return results + analysis."""
        with self.profile_block(f"function_{func.__name__}"):
            result = func(*args, **kwargs)
            analysis = self.analyze_memory_usage()
        
        return result, analysis
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage without full profiling."""
        snapshot = self.take_snapshot()
        
        return {
            'system_used_mb': snapshot.system_used_mb,
            'system_percent': snapshot.system_percent,
            'process_rss_mb': snapshot.process_rss_mb,
            'process_percent': snapshot.process_percent,
            'gpu_used_mb': snapshot.gpu_used_mb,
            'gpu_percent': snapshot.gpu_percent
        }
    
    def suggest_batch_size(self, current_batch_size: int, sample_memory_mb: float) -> Dict[str, Any]:
        """Suggest optimal batch size based on memory constraints."""
        current_usage = self.get_current_usage()
        
        suggestions = {
            'current_batch_size': current_batch_size,
            'current_memory_mb': current_usage['process_rss_mb'],
            'suggestions': {}
        }
        
        # System memory constraint
        available_system_mb = current_usage.get('system_used_mb', 0)
        if available_system_mb > 0:
            system_total = psutil.virtual_memory().total / (1024**2)
            available_system = system_total * 0.8 - available_system_mb  # Use 80% max
            
            if sample_memory_mb > 0:
                max_batch_system = int(available_system / sample_memory_mb)
                suggestions['suggestions']['system_memory'] = {
                    'max_batch_size': max(1, max_batch_system),
                    'reasoning': f'Based on {available_system:.0f}MB available system memory'
                }
        
        # GPU memory constraint
        if current_usage.get('gpu_used_mb', 0) > 0:
            gpu_total_mb = current_usage.get('gpu_total_mb', 0)
            gpu_used_mb = current_usage['gpu_used_mb']
            available_gpu = gpu_total_mb * 0.8 - gpu_used_mb  # Use 80% max
            
            if sample_memory_mb > 0 and available_gpu > 0:
                max_batch_gpu = int(available_gpu / sample_memory_mb)
                suggestions['suggestions']['gpu_memory'] = {
                    'max_batch_size': max(1, max_batch_gpu),
                    'reasoning': f'Based on {available_gpu:.0f}MB available GPU memory'
                }
        
        # Overall recommendation
        if suggestions['suggestions']:
            recommended_batch = min(
                suggestion['max_batch_size'] 
                for suggestion in suggestions['suggestions'].values()
            )
            suggestions['recommended_batch_size'] = max(1, min(recommended_batch, current_batch_size * 2))
        
        return suggestions


class MemoryMappedFile:
    """Utility for memory-mapped file access with optimizations."""
    
    def __init__(self, file_path: Union[str, Path], mode: str = 'r', access: int = mmap.ACCESS_READ):
        self.file_path = Path(file_path)
        self.mode = mode
        self.access = access
        
        self.file_handle: Optional[Any] = None
        self.memory_map: Optional[mmap.mmap] = None
        
        self._open_file()
    
    def _open_file(self):
        """Open file and create memory map."""
        try:
            if self.mode == 'r':
                self.file_handle = open(self.file_path, 'rb')
            else:
                self.file_handle = open(self.file_path, 'r+b')
            
            self.memory_map = mmap.mmap(
                self.file_handle.fileno(),
                0,
                access=self.access
            )
            
            logger.debug(f"Created memory map for: {self.file_path}")
            
        except Exception as e:
            logger.error(f"Failed to create memory map for {self.file_path}: {e}")
            self._cleanup()
            raise
    
    def read(self, size: int = -1, offset: int = 0) -> bytes:
        """Read data from memory-mapped file."""
        if self.memory_map is None:
            raise RuntimeError("Memory map not available")
        
        if offset > 0:
            self.memory_map.seek(offset)
        
        if size == -1:
            return self.memory_map.read()
        else:
            return self.memory_map.read(size)
    
    def readline(self, offset: int = 0) -> bytes:
        """Read a line from memory-mapped file."""
        if self.memory_map is None:
            raise RuntimeError("Memory map not available")
        
        if offset > 0:
            self.memory_map.seek(offset)
        
        return self.memory_map.readline()
    
    def find(self, pattern: bytes, start: int = 0) -> int:
        """Find pattern in memory-mapped file."""
        if self.memory_map is None:
            raise RuntimeError("Memory map not available")
        
        return self.memory_map.find(pattern, start)
    
    def size(self) -> int:
        """Get file size."""
        if self.memory_map is None:
            return 0
        return self.memory_map.size()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.memory_map:
            try:
                self.memory_map.close()
            except:
                pass
            self.memory_map = None
        
        if self.file_handle:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
    
    def __del__(self):
        self._cleanup()


def optimize_memory_usage():
    """Optimize current memory usage."""
    # Force garbage collection
    collected = gc.collect()
    
    # Clear PyTorch cache if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info(f"Memory optimization: collected {collected} objects")
    
    return {
        'gc_collected': collected,
        'torch_cache_cleared': TORCH_AVAILABLE and torch.cuda.is_available()
    }


def get_memory_info() -> Dict[str, Any]:
    """Get comprehensive memory information."""
    info = {}
    
    # System memory
    system_memory = psutil.virtual_memory()
    info['system'] = {
        'total_mb': system_memory.total / (1024**2),
        'available_mb': system_memory.available / (1024**2),
        'used_mb': system_memory.used / (1024**2),
        'percent': system_memory.percent
    }
    
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    info['process'] = {
        'rss_mb': process_memory.rss / (1024**2),
        'vms_mb': process_memory.vms / (1024**2),
        'percent': process.memory_percent()
    }
    
    # GPU memory
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info['gpu'] = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            
            info['gpu'][f'device_{i}'] = {
                'name': props.name,
                'total_mb': props.total_memory / (1024**2),
                'allocated_mb': allocated / (1024**2),
                'reserved_mb': reserved / (1024**2),
                'free_mb': (props.total_memory - reserved) / (1024**2)
            }
    
    # Python objects
    info['python'] = {
        'objects_count': len(gc.get_objects()),
        'gc_count': gc.get_count()
    }
    
    return info


@contextmanager
def memory_limit_context(limit_mb: float):
    """Context manager that monitors memory usage and warns if limit is exceeded."""
    profiler = MemoryProfiler()
    initial_usage = profiler.get_current_usage()['process_rss_mb']
    
    try:
        yield
    finally:
        final_usage = profiler.get_current_usage()['process_rss_mb']
        
        if final_usage > limit_mb:
            logger.warning(f"Memory limit exceeded: {final_usage:.1f}MB > {limit_mb:.1f}MB")
        
        growth = final_usage - initial_usage
        if growth > 0:
            logger.info(f"Memory growth: {growth:.1f}MB")


def estimate_tensor_memory(shape: tuple, dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory usage of a tensor in MB."""
    if not TORCH_AVAILABLE:
        return 0.0
    
    # Element size in bytes
    element_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1
    }
    
    element_size = element_sizes.get(dtype, 4)  # Default to 4 bytes
    total_elements = np.prod(shape)
    total_bytes = total_elements * element_size
    
    return total_bytes / (1024**2)  # Convert to MB


def suggest_memory_optimizations(current_usage: Dict[str, float]) -> List[str]:
    """Suggest memory optimizations based on current usage."""
    suggestions = []
    
    # System memory suggestions
    if current_usage.get('system_percent', 0) > 80:
        suggestions.append("High system memory usage. Consider:")
        suggestions.append("  - Reducing batch size")
        suggestions.append("  - Using memory mapping for large files")
        suggestions.append("  - Enabling data streaming instead of loading all data")
    
    # Process memory suggestions
    if current_usage.get('process_rss_mb', 0) > 2000:  # > 2GB
        suggestions.append("High process memory usage. Consider:")
        suggestions.append("  - Reducing cache sizes")
        suggestions.append("  - Using generator-based data loading")
        suggestions.append("  - Implementing data compression")
    
    # GPU memory suggestions
    if current_usage.get('gpu_percent', 0) > 80:
        suggestions.append("High GPU memory usage. Consider:")
        suggestions.append("  - Reducing batch size")
        suggestions.append("  - Using gradient checkpointing")
        suggestions.append("  - Enabling mixed precision training")
        suggestions.append("  - Using CPU offloading for some operations")
    
    if not suggestions:
        suggestions.append("Memory usage looks healthy!")
    
    return suggestions 