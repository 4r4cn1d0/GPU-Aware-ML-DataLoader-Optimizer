"""
System monitoring module for tracking CPU, memory, and I/O usage.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from collections import deque

import psutil


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics at a point in time."""
    
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    swap_used_mb: float
    swap_total_mb: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int
    timestamp: float = field(default_factory=time.time)


class SystemMonitor:
    """
    Monitor system resources (CPU, memory, I/O) during data loading operations.
    
    Features:
    - CPU utilization tracking
    - Memory usage monitoring
    - Disk I/O monitoring
    - Network I/O tracking
    - Process-specific monitoring
    - Load average tracking
    """
    
    def __init__(self, sample_interval: float = 0.1, track_process: bool = True):
        self.sample_interval = sample_interval
        self.track_process = track_process
        
        # Process tracking
        if track_process:
            self.process = psutil.Process()
            self.process_start_io = None
        else:
            self.process = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.start_time: Optional[float] = None
        self.start_io_counters: Optional[psutil._common.sdiskio] = None
        self.start_net_counters: Optional[psutil._common.snetio] = None
        
        logger.info("SystemMonitor initialized")
    
    def start_monitoring(self):
        """Start monitoring system metrics in a background thread."""
        if self.is_monitoring:
            logger.warning("System monitoring already active")
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.metrics_history.clear()
        
        # Record baseline I/O counters
        try:
            self.start_io_counters = psutil.disk_io_counters()
            self.start_net_counters = psutil.net_io_counters()
            
            if self.process:
                self.process_start_io = self.process.io_counters()
        except Exception as e:
            logger.warning(f"Failed to get baseline I/O counters: {e}")
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        if not self.is_monitoring:
            logger.warning("System monitoring not active")
            return {}
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        logger.info(f"Stopped system monitoring. Avg CPU: {stats.get('avg_cpu_percent', 0):.1f}%")
        
        return stats
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # I/O metrics (incremental from start)
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            if self.start_io_counters:
                current_io = psutil.disk_io_counters()
                if current_io:
                    disk_read_mb = (current_io.read_bytes - self.start_io_counters.read_bytes) / (1024**2)
                    disk_write_mb = (current_io.write_bytes - self.start_io_counters.write_bytes) / (1024**2)
            
            # Network metrics (incremental from start)
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            if self.start_net_counters:
                current_net = psutil.net_io_counters()
                if current_net:
                    network_sent_mb = (current_net.bytes_sent - self.start_net_counters.bytes_sent) / (1024**2)
                    network_recv_mb = (current_net.bytes_recv - self.start_net_counters.bytes_recv) / (1024**2)
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]  # Not available on Windows
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_used_mb=memory.used / (1024**2),
                memory_total_mb=memory.total / (1024**2),
                memory_percent=memory.percent,
                swap_used_mb=swap.used / (1024**2),
                swap_total_mb=swap.total / (1024**2),
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                load_average=load_avg,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                cpu_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                memory_percent=0.0,
                swap_used_mb=0.0,
                swap_total_mb=0.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0
            )
    
    def get_process_metrics(self) -> Dict[str, Any]:
        """Get current process-specific metrics."""
        if not self.process:
            return {}
        
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU info
            cpu_percent = self.process.cpu_percent()
            
            # I/O info
            io_info = {}
            if self.process_start_io:
                try:
                    current_io = self.process.io_counters()
                    io_info = {
                        'read_mb': (current_io.read_bytes - self.process_start_io.read_bytes) / (1024**2),
                        'write_mb': (current_io.write_bytes - self.process_start_io.write_bytes) / (1024**2),
                        'read_count': current_io.read_count - self.process_start_io.read_count,
                        'write_count': current_io.write_count - self.process_start_io.write_count
                    }
                except Exception:
                    pass
            
            # Thread info
            num_threads = self.process.num_threads()
            
            return {
                'pid': self.process.pid,
                'memory_rss_mb': memory_info.rss / (1024**2),
                'memory_vms_mb': memory_info.vms / (1024**2),
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'num_threads': num_threads,
                **io_info
            }
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return {}
    
    def get_memory_pressure(self) -> Dict[str, float]:
        """Get memory pressure indicators."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            pressure_score = 0.0
            
            # High memory usage
            if memory.percent > 80:
                pressure_score += (memory.percent - 80) / 20  # 0-1 scale
            
            # Swap usage
            if swap.total > 0 and swap.percent > 10:
                pressure_score += swap.percent / 100  # 0-1 scale
            
            # Memory availability
            available_gb = memory.available / (1024**3)
            if available_gb < 1.0:  # Less than 1GB available
                pressure_score += (1.0 - available_gb) / 1.0
            
            pressure_score = min(pressure_score, 1.0)  # Cap at 1.0
            
            return {
                'pressure_score': pressure_score,
                'memory_percent': memory.percent,
                'swap_percent': swap.percent,
                'available_gb': available_gb,
                'recommendation': self._get_memory_recommendation(pressure_score)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate memory pressure: {e}")
            return {'pressure_score': 0.0}
    
    def get_io_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze I/O patterns during monitoring."""
        if len(self.metrics_history) < 10:
            return {}
        
        # Extract I/O data
        read_values = [m.disk_read_mb for m in self.metrics_history]
        write_values = [m.disk_write_mb for m in self.metrics_history]
        
        # Calculate patterns
        total_read = max(read_values) if read_values else 0
        total_write = max(write_values) if write_values else 0
        
        # I/O intensity (MB/s)
        duration = (self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp)
        read_rate = total_read / duration if duration > 0 else 0
        write_rate = total_write / duration if duration > 0 else 0
        
        # I/O pattern classification
        if read_rate > 50:  # > 50 MB/s
            io_pattern = 'read_intensive'
        elif write_rate > 20:  # > 20 MB/s
            io_pattern = 'write_intensive'
        elif read_rate > 10 or write_rate > 5:
            io_pattern = 'moderate_io'
        else:
            io_pattern = 'light_io'
        
        return {
            'total_read_mb': total_read,
            'total_write_mb': total_write,
            'read_rate_mb_per_sec': read_rate,
            'write_rate_mb_per_sec': write_rate,
            'io_pattern': io_pattern,
            'read_write_ratio': read_rate / max(write_rate, 0.1)  # Avoid div by zero
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health and resource availability."""
        metrics = self.get_current_metrics()
        memory_pressure = self.get_memory_pressure()
        
        health_score = 1.0
        issues = []
        recommendations = []
        
        # CPU check
        if metrics.cpu_percent > 90:
            health_score -= 0.3
            issues.append("High CPU usage")
            recommendations.append("Consider reducing worker processes")
        elif metrics.cpu_percent > 70:
            health_score -= 0.1
            issues.append("Moderate CPU usage")
        
        # Memory check
        memory_pressure_score = memory_pressure.get('pressure_score', 0)
        if memory_pressure_score > 0.8:
            health_score -= 0.4
            issues.append("High memory pressure")
            recommendations.append("Reduce batch size or enable memory optimization")
        elif memory_pressure_score > 0.5:
            health_score -= 0.2
            issues.append("Moderate memory pressure")
        
        # Load average check (Unix-like systems)
        if metrics.load_average[0] > psutil.cpu_count() * 2:
            health_score -= 0.2
            issues.append("High system load")
            recommendations.append("System is overloaded")
        
        # I/O check
        if metrics.disk_read_mb > 1000 or metrics.disk_write_mb > 500:  # Very high I/O
            health_score -= 0.1
            issues.append("High disk I/O")
            recommendations.append("Consider using faster storage or caching")
        
        health_score = max(0.0, health_score)
        
        # Overall assessment
        if health_score > 0.8:
            status = "excellent"
        elif health_score > 0.6:
            status = "good"
        elif health_score > 0.4:
            status = "fair"
        else:
            status = "poor"
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'metrics_summary': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'load_average': metrics.load_average[0],
                'memory_pressure': memory_pressure_score
            }
        }
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics from monitoring history."""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_used_mb for m in self.metrics_history]
        memory_percent_values = [m.memory_percent for m in self.metrics_history]
        
        stats = {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'min_cpu_percent': min(cpu_values),
            'cpu_std': self._std(cpu_values) if len(cpu_values) > 1 else 0.0,
            
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'min_memory_mb': min(memory_values),
            'memory_std_mb': self._std(memory_values) if len(memory_values) > 1 else 0.0,
            
            'peak_memory_percent': max(memory_percent_values),
            'avg_memory_percent': sum(memory_percent_values) / len(memory_percent_values),
        }
        
        # I/O statistics
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            stats.update({
                'io_read_mb': final_metrics.disk_read_mb,
                'io_write_mb': final_metrics.disk_write_mb,
                'network_sent_mb': final_metrics.network_sent_mb,
                'network_recv_mb': final_metrics.network_recv_mb,
            })
            
            # Calculate I/O rates
            if self.start_time:
                duration = time.time() - self.start_time
                stats.update({
                    'io_read_rate_mb_per_sec': final_metrics.disk_read_mb / duration if duration > 0 else 0,
                    'io_write_rate_mb_per_sec': final_metrics.disk_write_mb / duration if duration > 0 else 0,
                })
        
        # Cache hit rate estimation (very rough heuristic)
        if 'io_read_mb' in stats and 'io_write_mb' in stats:
            total_io = stats['io_read_mb'] + stats['io_write_mb']
            if total_io > 0:
                # Assume lower I/O relative to data processed indicates better caching
                # This is a very rough estimate and would need refinement for real use
                estimated_cache_hit_rate = min(0.9, max(0.0, 1.0 - (total_io / 1000)))  # Rough heuristic
                stats['cache_hit_rate'] = estimated_cache_hit_rate
            else:
                stats['cache_hit_rate'] = 0.0
        else:
            stats['cache_hit_rate'] = 0.0
        
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
    
    def _get_memory_recommendation(self, pressure_score: float) -> str:
        """Get memory optimization recommendation."""
        if pressure_score > 0.8:
            return "Critical: Reduce batch size immediately"
        elif pressure_score > 0.6:
            return "High: Consider reducing batch size"
        elif pressure_score > 0.4:
            return "Moderate: Monitor memory usage"
        else:
            return "Good: Memory usage is healthy"
    
    def export_metrics_history(self) -> List[Dict[str, Any]]:
        """Export the full metrics history."""
        return [
            {
                'timestamp': m.timestamp,
                'cpu_percent': m.cpu_percent,
                'memory_used_mb': m.memory_used_mb,
                'memory_percent': m.memory_percent,
                'disk_read_mb': m.disk_read_mb,
                'disk_write_mb': m.disk_write_mb,
                'network_sent_mb': m.network_sent_mb,
                'network_recv_mb': m.network_recv_mb,
                'load_average': m.load_average,
                'process_count': m.process_count
            }
            for m in self.metrics_history
        ]
    
    def get_resource_timeline(self) -> Dict[str, List[float]]:
        """Get resource usage timeline for visualization."""
        if not self.metrics_history:
            return {'timestamps': [], 'cpu_percent': [], 'memory_mb': [], 'io_read_mb': []}
        
        start_time = self.metrics_history[0].timestamp
        
        return {
            'timestamps': [(m.timestamp - start_time) for m in self.metrics_history],
            'cpu_percent': [m.cpu_percent for m in self.metrics_history],
            'memory_mb': [m.memory_used_mb for m in self.metrics_history],
            'io_read_mb': [m.disk_read_mb for m in self.metrics_history],
            'io_write_mb': [m.disk_write_mb for m in self.metrics_history]
        } 