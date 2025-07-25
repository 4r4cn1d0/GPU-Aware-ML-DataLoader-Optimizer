"""
Base optimized data loader with common GPU-aware optimizations.
"""

import os
import mmap
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator, Tuple, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class OptimizedDataset(Dataset, ABC):
    """Base class for optimized datasets with memory-mapped I/O and caching."""
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        prefetch_size: int = 100
    ):
        self.data_paths = [Path(p) for p in (data_path if isinstance(data_path, list) else [data_path])]
        self.use_memory_mapping = use_memory_mapping
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        
        # Memory mapped files
        self.memory_maps: Dict[Path, mmap.mmap] = {}
        self.file_handles: Dict[Path, Any] = {}  # Keep file handles open
        
        # Caching
        self.cache: Dict[int, Any] = {}
        self.cache_access_order: List[int] = []
        self.cache_lock = threading.Lock()
        
        # Prefetch queue
        self.prefetch_queue: Queue = Queue(maxsize=prefetch_size)
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_enabled = False
        
        # Initialize
        self._initialize()
        
        logger.info(f"OptimizedDataset initialized: {len(self.data_paths)} files, "
                   f"cache_size={cache_size}, memory_mapping={use_memory_mapping}")
    
    def _initialize(self):
        """Initialize memory mapping and prepare datasets."""
        for path in self.data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            if self.use_memory_mapping:
                self._create_memory_map(path)
    
    def _create_memory_map(self, path: Path):
        """Create memory mapping for a file."""
        try:
            file_handle = open(path, 'rb')
            memory_map = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
            self.file_handles[path] = file_handle
            self.memory_maps[path] = memory_map
            
            logger.debug(f"Created memory map for: {path}")
            
        except Exception as e:
            logger.warning(f"Failed to create memory map for {path}: {e}")
            # Fallback to regular file I/O
            if path in self.file_handles:
                self.file_handles[path].close()
                del self.file_handles[path]
    
    def enable_prefetch(self):
        """Enable background prefetching."""
        if self.prefetch_enabled:
            return
        
        self.prefetch_enabled = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        logger.debug("Enabled prefetching")
    
    def disable_prefetch(self):
        """Disable background prefetching."""
        self.prefetch_enabled = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)
    
    def _prefetch_worker(self):
        """Background worker for prefetching data."""
        while self.prefetch_enabled:
            try:
                # This is a placeholder - subclasses should implement actual prefetching
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                time.sleep(1.0)
    
    def _get_from_cache(self, index: int) -> Optional[Any]:
        """Get item from cache."""
        with self.cache_lock:
            if index in self.cache:
                # Update access order (LRU)
                self.cache_access_order.remove(index)
                self.cache_access_order.append(index)
                return self.cache[index]
        return None
    
    def _put_in_cache(self, index: int, item: Any):
        """Put item in cache with LRU eviction."""
        with self.cache_lock:
            # Remove if already exists
            if index in self.cache:
                self.cache_access_order.remove(index)
            
            # Add new item
            self.cache[index] = item
            self.cache_access_order.append(index)
            
            # Evict if cache is full
            while len(self.cache) > self.cache_size:
                oldest_index = self.cache_access_order.pop(0)
                del self.cache[oldest_index]
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod 
    def __getitem__(self, index: int) -> Any:
        """Get a sample from the dataset."""
        pass
    
    def __del__(self):
        """Cleanup resources."""
        self.disable_prefetch()
        
        # Close memory maps
        for mmap_obj in self.memory_maps.values():
            try:
                mmap_obj.close()
            except:
                pass
        
        # Close file handles
        for file_handle in self.file_handles.values():
            try:
                file_handle.close()
            except:
                pass


class BaseOptimizedLoader:
    """
    Base class for GPU-aware optimized data loaders.
    
    Features:
    - Memory-mapped I/O for large files
    - Pinned memory for faster GPU transfers
    - Zero-copy loading where possible
    - Intelligent prefetching
    - Dynamic batch size adjustment
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        enable_zero_copy: bool = True,
        **kwargs
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.use_memory_mapping = use_memory_mapping
        self.cache_size = cache_size
        self.enable_zero_copy = enable_zero_copy
        self.kwargs = kwargs
        
        # Dataset and loader
        self.dataset: Optional[OptimizedDataset] = None
        self.dataloader: Optional[DataLoader] = None
        
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Statistics
        self.stats = {
            'batches_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'gpu_transfers': 0,
            'zero_copy_operations': 0
        }
        
        logger.info(f"BaseOptimizedLoader initialized: batch_size={batch_size}, "
                   f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    @abstractmethod
    def _create_dataset(self) -> OptimizedDataset:
        """Create the optimized dataset. Must be implemented by subclasses."""
        pass
    
    def _create_dataloader(self) -> DataLoader:
        """Create the PyTorch DataLoader with optimizations."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        # Custom collate function for zero-copy operations
        collate_fn = self._zero_copy_collate_fn if self.enable_zero_copy else None
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Shuffling handled at dataset level for efficiency
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def _zero_copy_collate_fn(self, batch: List[Any]) -> Any:
        """Custom collate function that attempts zero-copy operations."""
        try:
            if not batch:
                return batch
            
            # Attempt to stack tensors without copying
            if all(torch.is_tensor(item) for item in batch):
                # Use torch.stack for zero-copy when possible
                result = torch.stack(batch, dim=0)
                self.stats['zero_copy_operations'] += 1
                return result
            
            # Handle tuples/lists of tensors
            if all(isinstance(item, (tuple, list)) for item in batch):
                if all(len(item) == len(batch[0]) for item in batch):
                    # Transpose and stack each element
                    transposed = list(zip(*batch))
                    result = []
                    for elements in transposed:
                        if all(torch.is_tensor(elem) for elem in elements):
                            stacked = torch.stack(list(elements), dim=0)
                            result.append(stacked)
                            self.stats['zero_copy_operations'] += 1
                        else:
                            result.append(list(elements))
                    return tuple(result) if isinstance(batch[0], tuple) else result
            
            # Fallback to default collation
            return torch.utils.data.dataloader.default_collate(batch)
            
        except Exception as e:
            logger.debug(f"Zero-copy collation failed, falling back to default: {e}")
            return torch.utils.data.dataloader.default_collate(batch)
    
    def get_dataloader(self) -> DataLoader:
        """Get the configured DataLoader."""
        if self.dataloader is None:
            self.dataloader = self._create_dataloader()
        return self.dataloader
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches with GPU optimizations."""
        dataloader = self.get_dataloader()
        
        for batch in dataloader:
            self.stats['batches_loaded'] += 1
            
            # Transfer to GPU with optimization
            if self.device.type == 'cuda' and not self._is_on_gpu(batch):
                batch = self._transfer_to_gpu(batch)
                self.stats['gpu_transfers'] += 1
            
            yield batch
    
    def _is_on_gpu(self, batch: Any) -> bool:
        """Check if batch is already on GPU."""
        if torch.is_tensor(batch):
            return batch.device.type == 'cuda'
        elif isinstance(batch, (list, tuple)):
            return all(self._is_on_gpu(item) for item in batch)
        elif isinstance(batch, dict):
            return all(self._is_on_gpu(value) for value in batch.values())
        else:
            return False
    
    def _transfer_to_gpu(self, batch: Any) -> Any:
        """Transfer batch to GPU with optimizations."""
        if not torch.cuda.is_available():
            return batch
        
        # Use CUDA stream for asynchronous transfer
        if self.cuda_stream:
            with torch.cuda.stream(self.cuda_stream):
                return self._recursive_gpu_transfer(batch, non_blocking=True)
        else:
            return self._recursive_gpu_transfer(batch, non_blocking=False)
    
    def _recursive_gpu_transfer(self, obj: Any, non_blocking: bool = True) -> Any:
        """Recursively transfer objects to GPU."""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=non_blocking)
        elif isinstance(obj, list):
            return [self._recursive_gpu_transfer(item, non_blocking) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._recursive_gpu_transfer(item, non_blocking) for item in obj)
        elif isinstance(obj, dict):
            return {key: self._recursive_gpu_transfer(value, non_blocking) 
                   for key, value in obj.items()}
        else:
            return obj
    
    def adjust_batch_size(self, new_batch_size: int):
        """Dynamically adjust batch size."""
        if new_batch_size == self.batch_size:
            return
        
        logger.info(f"Adjusting batch size: {self.batch_size} -> {new_batch_size}")
        
        self.batch_size = new_batch_size
        
        # Recreate dataloader with new batch size
        self.dataloader = None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        usage = {
            'dataset_cache_size': len(self.dataset.cache) if self.dataset else 0,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'zero_copy_rate': self.stats['zero_copy_operations'] / max(self.stats['batches_loaded'], 1),
            'gpu_transfer_rate': self.stats['gpu_transfers'] / max(self.stats['batches_loaded'], 1)
        }
        
        # Add GPU memory usage if available
        if torch.cuda.is_available():
            usage.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2)
            })
        
        return usage
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loader statistics."""
        stats = self.stats.copy()
        
        # Add derived metrics
        total_operations = stats['batches_loaded']
        if total_operations > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            stats['zero_copy_rate'] = stats['zero_copy_operations'] / total_operations
            stats['gpu_transfer_rate'] = stats['gpu_transfers'] / total_operations
        
        # Add memory usage
        stats.update(self.get_memory_usage())
        
        return stats
    
    def optimize_for_gpu_memory(self, available_memory_mb: float) -> Dict[str, Any]:
        """Optimize configuration based on available GPU memory."""
        recommendations = {}
        
        if not torch.cuda.is_available():
            return recommendations
        
        # Estimate memory per batch
        try:
            # Get a sample batch to estimate memory usage
            sample_batch = next(iter(self))
            batch_memory_mb = self._estimate_batch_memory(sample_batch)
            
            # Calculate optimal batch size
            safety_factor = 0.8  # Use 80% of available memory
            max_batch_size = int((available_memory_mb * safety_factor) / batch_memory_mb)
            max_batch_size = max(1, min(max_batch_size, 512))  # Reasonable bounds
            
            if max_batch_size != self.batch_size:
                recommendations['suggested_batch_size'] = max_batch_size
                recommendations['current_batch_size'] = self.batch_size
                recommendations['estimated_memory_per_batch_mb'] = batch_memory_mb
                recommendations['reason'] = f"Current batch uses ~{batch_memory_mb:.1f}MB, " \
                                         f"GPU has {available_memory_mb:.1f}MB available"
        
        except Exception as e:
            logger.warning(f"Failed to optimize for GPU memory: {e}")
        
        return recommendations
    
    def _estimate_batch_memory(self, batch: Any) -> float:
        """Estimate memory usage of a batch in MB."""
        try:
            total_bytes = 0
            
            def count_tensor_bytes(tensor):
                if torch.is_tensor(tensor):
                    return tensor.numel() * tensor.element_size()
                return 0
            
            total_bytes = self._recursive_count_bytes(batch, count_tensor_bytes)
            return total_bytes / (1024**2)
            
        except Exception as e:
            logger.warning(f"Failed to estimate batch memory: {e}")
            return 1.0  # Default estimate
    
    def _recursive_count_bytes(self, obj: Any, count_func: callable) -> int:
        """Recursively count bytes in nested structures."""
        if torch.is_tensor(obj):
            return count_func(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._recursive_count_bytes(item, count_func) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._recursive_count_bytes(value, count_func) for value in obj.values())
        else:
            return 0
    
    def warmup(self, num_batches: int = 5):
        """Warmup the loader by prefetching several batches."""
        logger.info(f"Warming up loader with {num_batches} batches...")
        
        start_time = time.time()
        
        for i, batch in enumerate(self):
            if i >= num_batches:
                break
            
            # Just iterate to trigger prefetching and caching
            pass
        
        warmup_time = time.time() - start_time
        logger.info(f"Warmup completed in {warmup_time:.2f}s")
        
        return {
            'warmup_time_sec': warmup_time,
            'batches_warmed': min(num_batches, i + 1),
            'avg_batch_time_sec': warmup_time / min(num_batches, i + 1)
        }
    
    def __len__(self) -> int:
        """Get number of batches."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        dataset_length = len(self.dataset)
        if self.drop_last:
            return dataset_length // self.batch_size
        else:
            return (dataset_length + self.batch_size - 1) // self.batch_size 