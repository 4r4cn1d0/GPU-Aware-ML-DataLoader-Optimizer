"""
GPU monitoring module for tracking memory usage and utilization.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from collections import deque

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU metrics at a point in time."""
    
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    utilization_percent: float
    temperature_c: float
    power_draw_w: float
    timestamp: float = field(default_factory=time.time)


class GPUMonitor:
    """
    Monitor GPU memory usage and utilization during data loading operations.
    
    Features:
    - Real-time GPU memory tracking
    - Utilization monitoring
    - Temperature and power monitoring
    - Peak and average statistics
    - Multi-GPU support
    """
    
    def __init__(self, gpu_id: int = 0, sample_interval: float = 0.1):
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        
        # Initialize NVML if available
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.nvml_initialized = True
                logger.info(f"NVML initialized for GPU {gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k samples
        
        # Statistics
        self.peak_memory_mb = 0.0
        self.peak_utilization = 0.0
        self.start_time: Optional[float] = None
        
        logger.info(f"GPUMonitor initialized for GPU {gpu_id}")
    
    def start_monitoring(self):
        """Start monitoring GPU metrics in a background thread."""
        if self.is_monitoring:
            logger.warning("GPU monitoring already active")
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.metrics_history.clear()
        self.peak_memory_mb = 0.0
        self.peak_utilization = 0.0
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started GPU monitoring")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        if not self.is_monitoring:
            logger.warning("GPU monitoring not active")
            return {}
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        logger.info(f"Stopped GPU monitoring. Peak memory: {stats.get('peak_memory_mb', 0):.1f} MB")
        
        return stats
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        try:
            if self.nvml_initialized:
                return self._get_nvml_metrics()
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                return self._get_torch_metrics()
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to get current GPU metrics: {e}")
            return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        metrics = self.get_current_metrics()
        if metrics:
            return {
                'used_mb': metrics.memory_used_mb,
                'total_mb': metrics.memory_total_mb,
                'percent': metrics.memory_percent,
                'available_mb': metrics.memory_total_mb - metrics.memory_used_mb
            }
        return {}
    
    def check_memory_constraint(self, required_mb: float, buffer_percent: float = 0.1) -> bool:
        """Check if required memory fits within GPU memory constraints."""
        metrics = self.get_current_metrics()
        if not metrics:
            logger.warning("Cannot check memory constraint - no GPU metrics available")
            return True  # Assume it's fine if we can't measure
        
        available_mb = metrics.memory_total_mb - metrics.memory_used_mb
        required_with_buffer = required_mb * (1 + buffer_percent)
        
        return available_mb >= required_with_buffer
    
    def suggest_max_batch_size(self, sample_memory_mb: float, overhead_mb: float = 500) -> int:
        """Suggest maximum batch size based on available GPU memory."""
        metrics = self.get_current_metrics()
        if not metrics:
            return 32  # Default fallback
        
        available_mb = metrics.memory_total_mb - metrics.memory_used_mb - overhead_mb
        
        if sample_memory_mb <= 0:
            return 32  # Fallback
        
        max_batch_size = int(available_mb / sample_memory_mb)
        
        # Reasonable bounds
        max_batch_size = max(1, min(max_batch_size, 1024))
        
        return max_batch_size
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # Update peaks
                    self.peak_memory_mb = max(self.peak_memory_mb, metrics.memory_used_mb)
                    self.peak_utilization = max(self.peak_utilization, metrics.utilization_percent)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _get_nvml_metrics(self) -> GPUMetrics:
        """Get metrics using NVML (most accurate)."""
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        memory_used_mb = mem_info.used / (1024**2)
        memory_total_mb = mem_info.total / (1024**2)
        memory_percent = (mem_info.used / mem_info.total) * 100
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        utilization_percent = util.gpu
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temp = 0.0
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
        except:
            power = 0.0
        
        return GPUMetrics(
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_percent=memory_percent,
            utilization_percent=utilization_percent,
            temperature_c=temp,
            power_draw_w=power
        )
    
    def _get_torch_metrics(self) -> GPUMetrics:
        """Get metrics using PyTorch (fallback)."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        # Memory info
        memory_used = torch.cuda.memory_allocated(self.gpu_id)
        memory_reserved = torch.cuda.memory_reserved(self.gpu_id)
        
        # Try to get total memory
        try:
            props = torch.cuda.get_device_properties(self.gpu_id)
            memory_total = props.total_memory
        except:
            memory_total = memory_reserved * 2  # Rough estimate
        
        memory_used_mb = memory_used / (1024**2)
        memory_total_mb = memory_total / (1024**2)
        memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
        
        return GPUMetrics(
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_percent=memory_percent,
            utilization_percent=0.0,  # Not available via PyTorch
            temperature_c=0.0,
            power_draw_w=0.0
        )
    
    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics from monitoring history."""
        if not self.metrics_history:
            return {}
        
        memory_values = [m.memory_used_mb for m in self.metrics_history]
        util_values = [m.utilization_percent for m in self.metrics_history if m.utilization_percent > 0]
        
        stats = {
            'peak_memory_mb': max(memory_values) if memory_values else 0.0,
            'avg_memory_mb': sum(memory_values) / len(memory_values) if memory_values else 0.0,
            'min_memory_mb': min(memory_values) if memory_values else 0.0,
            'memory_std_mb': self._std(memory_values) if len(memory_values) > 1 else 0.0,
        }
        
        if util_values:
            stats.update({
                'peak_utilization': max(util_values),
                'avg_utilization': sum(util_values) / len(util_values),
                'min_utilization': min(util_values),
                'utilization_std': self._std(util_values) if len(util_values) > 1 else 0.0,
            })
        else:
            stats.update({
                'peak_utilization': 0.0,
                'avg_utilization': 0.0,
                'min_utilization': 0.0,
                'utilization_std': 0.0,
            })
        
        # Temporal statistics
        if self.start_time:
            duration = time.time() - self.start_time
            stats['monitoring_duration_sec'] = duration
            stats['samples_collected'] = len(self.metrics_history)
            stats['sampling_rate_hz'] = len(self.metrics_history) / duration if duration > 0 else 0
        
        return stats
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_metrics_history(self) -> List[Dict[str, Any]]:
        """Export the full metrics history."""
        return [
            {
                'timestamp': m.timestamp,
                'memory_used_mb': m.memory_used_mb,
                'memory_total_mb': m.memory_total_mb,
                'memory_percent': m.memory_percent,
                'utilization_percent': m.utilization_percent,
                'temperature_c': m.temperature_c,
                'power_draw_w': m.power_draw_w
            }
            for m in self.metrics_history
        ]
    
    def get_memory_timeline(self) -> Dict[str, List[float]]:
        """Get memory usage timeline for visualization."""
        if not self.metrics_history:
            return {'timestamps': [], 'memory_mb': [], 'utilization': []}
        
        start_time = self.metrics_history[0].timestamp
        
        return {
            'timestamps': [(m.timestamp - start_time) for m in self.metrics_history],
            'memory_mb': [m.memory_used_mb for m in self.metrics_history],
            'utilization': [m.utilization_percent for m in self.metrics_history]
        }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get information about available GPUs."""
        gpu_info = []
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info.append({
                        'id': i,
                        'name': name,
                        'memory_total_mb': mem_info.total / (1024**2),
                        'memory_used_mb': mem_info.used / (1024**2),
                        'memory_free_mb': mem_info.free / (1024**2)
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU info via NVML: {e}")
        
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_used = torch.cuda.memory_allocated(i)
                    memory_total = props.total_memory
                    
                    gpu_info.append({
                        'id': i,
                        'name': props.name,
                        'memory_total_mb': memory_total / (1024**2),
                        'memory_used_mb': memory_used / (1024**2),
                        'memory_free_mb': (memory_total - memory_used) / (1024**2)
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU info via PyTorch: {e}")
        
        return gpu_info
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.is_monitoring:
            self.stop_monitoring()
        
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass 