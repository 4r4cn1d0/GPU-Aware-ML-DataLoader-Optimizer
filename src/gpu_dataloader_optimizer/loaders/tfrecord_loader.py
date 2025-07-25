"""
Optimized TFRecord data loader with GPU-aware optimizations.
"""

import os
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import logging

import numpy as np
import torch

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from .base_loader import BaseOptimizedLoader, OptimizedDataset


logger = logging.getLogger(__name__)


class TFRecordDataset(OptimizedDataset):
    """Optimized TFRecord dataset with memory mapping and caching."""
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        feature_description: Dict[str, Any],
        transform: Optional[Callable] = None,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        prefetch_size: int = 100,
        compression_type: str = 'AUTO'
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFRecord loading. Install with: pip install tensorflow")
        
        self.feature_description = feature_description
        self.transform = transform
        self.compression_type = compression_type
        
        # TensorFlow dataset
        self.tf_dataset = None
        self.tf_iterator = None
        
        # Sample counting
        self._length: Optional[int] = None
        
        super().__init__(data_path, use_memory_mapping, cache_size, prefetch_size)
        
        # Create TensorFlow dataset
        self._create_tf_dataset()
        
        logger.info(f"TFRecordDataset initialized: {len(self.data_paths)} files")
    
    def _create_tf_dataset(self):
        """Create TensorFlow dataset for TFRecord files."""
        # Convert paths to strings
        file_paths = [str(path) for path in self.data_paths]
        
        # Create dataset
        self.tf_dataset = tf.data.TFRecordDataset(
            file_paths,
            compression_type=self.compression_type,
            num_parallel_reads=min(4, len(file_paths))
        )
        
        # Parse records
        self.tf_dataset = self.tf_dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Prefetch for performance
        self.tf_dataset = self.tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    def _parse_function(self, example_proto):
        """Parse a single TFRecord example."""
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    def _count_records(self) -> int:
        """Count total number of records across all files."""
        if self._length is not None:
            return self._length
        
        total_count = 0
        
        for path in self.data_paths:
            try:
                # Use TensorFlow to count records
                dataset = tf.data.TFRecordDataset(str(path), compression_type=self.compression_type)
                count = sum(1 for _ in dataset)
                total_count += count
                logger.debug(f"File {path}: {count} records")
                
            except Exception as e:
                logger.warning(f"Failed to count records in {path}: {e}")
                # Fallback estimation
                file_size = path.stat().st_size
                estimated_count = max(1, file_size // 1024)  # Rough estimate
                total_count += estimated_count
        
        self._length = total_count
        logger.info(f"Total TFRecord count: {total_count}")
        
        return total_count
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._count_records()
    
    def __getitem__(self, index: int) -> Any:
        """Get a sample from the dataset."""
        # Check cache first
        cached_item = self._get_from_cache(index)
        if cached_item is not None:
            return cached_item
        
        try:
            # Create iterator if needed
            if self.tf_iterator is None:
                self.tf_iterator = iter(self.tf_dataset)
            
            # Skip to the desired index
            # Note: This is inefficient for random access. In practice, 
            # TFRecords are typically used sequentially
            current_index = getattr(self, '_current_index', 0)
            
            if index < current_index:
                # Reset iterator if we need to go backwards
                self.tf_iterator = iter(self.tf_dataset)
                current_index = 0
            
            # Skip forward to the desired index
            while current_index < index:
                next(self.tf_iterator)
                current_index += 1
            
            # Get the item
            tf_item = next(self.tf_iterator)
            self._current_index = current_index + 1
            
            # Convert TensorFlow tensors to PyTorch tensors
            item = self._tf_to_torch(tf_item)
            
            # Apply transform if provided
            if self.transform:
                item = self.transform(item)
            
            # Cache the item
            self._put_in_cache(index, item)
            
            return item
            
        except (StopIteration, tf.errors.OutOfRangeError):
            raise IndexError(f"Index {index} out of range")
        
        except Exception as e:
            logger.error(f"Failed to get item {index}: {e}")
            raise
    
    def _tf_to_torch(self, tf_item: Dict[str, tf.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert TensorFlow tensors to PyTorch tensors."""
        torch_item = {}
        
        for key, tf_tensor in tf_item.items():
            try:
                # Convert to numpy first
                numpy_array = tf_tensor.numpy()
                
                # Handle different data types
                if numpy_array.dtype == np.object_:
                    # Handle string/bytes data
                    if isinstance(numpy_array.item(), bytes):
                        # Decode bytes to string
                        torch_item[key] = numpy_array.item().decode('utf-8')
                    else:
                        torch_item[key] = numpy_array.item()
                else:
                    # Convert to PyTorch tensor
                    torch_tensor = torch.from_numpy(numpy_array)
                    torch_item[key] = torch_tensor
                    
            except Exception as e:
                logger.warning(f"Failed to convert tensor {key}: {e}")
                torch_item[key] = tf_tensor  # Keep as TensorFlow tensor
        
        return torch_item
    
    def get_sequential_iterator(self):
        """Get an iterator for sequential access (more efficient than random access)."""
        return iter(self.tf_dataset)


class TFRecordLoader(BaseOptimizedLoader):
    """
    GPU-aware optimized TFRecord data loader.
    
    Features:
    - Memory-mapped I/O for large TFRecord files
    - Efficient TensorFlow to PyTorch tensor conversion
    - Parallel record parsing
    - Intelligent caching and prefetching
    - Zero-copy operations where possible
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        feature_description: Dict[str, Any],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        enable_zero_copy: bool = True,
        compression_type: str = 'AUTO',
        transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize TFRecord loader.
        
        Args:
            data_path: Path(s) to TFRecord file(s)
            feature_description: TensorFlow feature description for parsing
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Use pinned memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker
            use_memory_mapping: Use memory-mapped I/O
            cache_size: Size of LRU cache
            enable_zero_copy: Enable zero-copy operations
            compression_type: TFRecord compression ('AUTO', 'GZIP', 'ZLIB', '')
            transform: Optional transform function
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TFRecord loading. Install with: pip install tensorflow")
        
        self.feature_description = feature_description
        self.compression_type = compression_type
        self.transform = transform
        
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            use_memory_mapping=use_memory_mapping,
            cache_size=cache_size,
            enable_zero_copy=enable_zero_copy,
            **kwargs
        )
        
        logger.info(f"TFRecordLoader initialized for {len(self.data_paths) if isinstance(data_path, list) else 1} files")
    
    def _create_dataset(self) -> TFRecordDataset:
        """Create the optimized TFRecord dataset."""
        return TFRecordDataset(
            data_path=self.data_path,
            feature_description=self.feature_description,
            transform=self.transform,
            use_memory_mapping=self.use_memory_mapping,
            cache_size=self.cache_size,
            prefetch_size=self.prefetch_factor * self.batch_size,
            compression_type=self.compression_type
        )
    
    def create_feature_description_from_example(self, example_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Automatically create feature description from a sample TFRecord file.
        
        This is a helper method to infer the feature description from the first record.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        example_path = Path(example_path)
        if not example_path.exists():
            raise FileNotFoundError(f"Example file not found: {example_path}")
        
        try:
            # Read first record
            dataset = tf.data.TFRecordDataset(str(example_path), compression_type=self.compression_type)
            first_record = next(iter(dataset))
            
            # Parse without feature description to inspect
            example = tf.train.Example()
            example.ParseFromString(first_record.numpy())
            
            feature_description = {}
            
            for key, feature in example.features.feature.items():
                if feature.HasField('bytes_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.string)
                elif feature.HasField('float_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
                elif feature.HasField('int64_list'):
                    feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
                else:
                    logger.warning(f"Unknown feature type for key: {key}")
                    feature_description[key] = tf.io.FixedLenFeature([], tf.string)
            
            logger.info(f"Inferred feature description: {feature_description}")
            return feature_description
            
        except Exception as e:
            logger.error(f"Failed to infer feature description: {e}")
            raise
    
    def get_tf_dataset_for_benchmarking(self) -> tf.data.Dataset:
        """Get the underlying TensorFlow dataset for benchmarking."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        return self.dataset.tf_dataset.batch(self.batch_size)
    
    def optimize_tf_pipeline(self) -> Dict[str, Any]:
        """Optimize the TensorFlow pipeline for better performance."""
        optimizations = {
            'applied_optimizations': [],
            'recommendations': []
        }
        
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        # Apply TensorFlow optimizations
        original_dataset = self.dataset.tf_dataset
        
        # 1. Parallel interleave for multiple files
        if len(self.data_paths) > 1:
            file_paths = [str(path) for path in self.data_paths]
            optimized_dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            optimized_dataset = optimized_dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename, compression_type=self.compression_type),
                cycle_length=min(4, len(file_paths)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            optimized_dataset = optimized_dataset.map(
                self.dataset._parse_function,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            optimizations['applied_optimizations'].append('parallel_interleave')
        else:
            optimized_dataset = original_dataset
        
        # 2. Cache if dataset is small enough
        try:
            # Estimate dataset size
            sample_size = 0
            count = 0
            for item in optimized_dataset.take(10):
                sample_size += len(tf.io.serialize_tensor(item))
                count += 1
            
            if count > 0:
                avg_size = sample_size / count
                estimated_total_size = avg_size * len(self.dataset)
                
                # Cache if dataset is smaller than 1GB
                if estimated_total_size < 1024**3:
                    optimized_dataset = optimized_dataset.cache()
                    optimizations['applied_optimizations'].append('caching')
                else:
                    optimizations['recommendations'].append('Dataset too large for caching')
        
        except Exception as e:
            logger.warning(f"Failed to apply caching optimization: {e}")
        
        # 3. Prefetch
        optimized_dataset = optimized_dataset.prefetch(tf.data.AUTOTUNE)
        optimizations['applied_optimizations'].append('prefetch')
        
        # Update the dataset
        self.dataset.tf_dataset = optimized_dataset
        
        return optimizations
    
    @staticmethod
    def create_simple_feature_description(
        image_shape: Optional[tuple] = None,
        label_type: str = 'int64',
        include_filename: bool = False
    ) -> Dict[str, Any]:
        """
        Create a simple feature description for common use cases.
        
        Args:
            image_shape: Shape of image data (height, width, channels)
            label_type: Type of label ('int64', 'float32')
            include_filename: Whether to include filename field
        
        Returns:
            Feature description dictionary
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        feature_desc = {}
        
        if image_shape:
            # Fixed-length image feature
            feature_desc['image'] = tf.io.FixedLenFeature(
                shape=image_shape, 
                dtype=tf.float32
            )
        else:
            # Variable-length raw image bytes
            feature_desc['image_raw'] = tf.io.FixedLenFeature([], tf.string)
        
        # Label
        if label_type == 'int64':
            feature_desc['label'] = tf.io.FixedLenFeature([], tf.int64)
        elif label_type == 'float32':
            feature_desc['label'] = tf.io.FixedLenFeature([], tf.float32)
        else:
            feature_desc['label'] = tf.io.FixedLenFeature([], tf.string)
        
        # Optional filename
        if include_filename:
            feature_desc['filename'] = tf.io.FixedLenFeature([], tf.string)
        
        return feature_desc
    
    def benchmark_parsing_performance(self, num_samples: int = 1000) -> Dict[str, float]:
        """Benchmark TFRecord parsing performance."""
        import time
        
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        # Benchmark TensorFlow parsing
        tf_start = time.time()
        tf_count = 0
        
        for item in self.dataset.tf_dataset.take(num_samples):
            tf_count += 1
        
        tf_time = time.time() - tf_start
        
        # Benchmark PyTorch conversion
        torch_start = time.time()
        torch_count = 0
        
        for i in range(min(num_samples, len(self.dataset))):
            item = self.dataset[i]
            torch_count += 1
        
        torch_time = time.time() - torch_start
        
        return {
            'tf_parsing_time_sec': tf_time,
            'tf_samples_per_sec': tf_count / tf_time if tf_time > 0 else 0,
            'torch_conversion_time_sec': torch_time,
            'torch_samples_per_sec': torch_count / torch_time if torch_time > 0 else 0,
            'total_time_sec': tf_time + torch_time,
            'total_samples_per_sec': (tf_count + torch_count) / (tf_time + torch_time) if (tf_time + torch_time) > 0 else 0
        } 