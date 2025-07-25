"""
Optimized CSV data loader with GPU-aware optimizations.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .base_loader import BaseOptimizedLoader, OptimizedDataset


logger = logging.getLogger(__name__)


class CSVDataset(OptimizedDataset):
    """Optimized CSV dataset with memory mapping and chunked loading."""
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        prefetch_size: int = 100,
        chunk_size: int = 10000,
        use_dask: bool = True,
        dtype_dict: Optional[Dict[str, str]] = None,
        **pandas_kwargs
    ):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.chunk_size = chunk_size
        self.use_dask = use_dask and DASK_AVAILABLE
        self.dtype_dict = dtype_dict or {}
        self.pandas_kwargs = pandas_kwargs
        
        # DataFrame storage
        self.dataframes: List[pd.DataFrame] = []
        self.dask_df: Optional[dd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
        
        # Index mapping for multiple files
        self.file_indices: List[Tuple[int, int]] = []  # (start_idx, end_idx) for each file
        self._length: Optional[int] = None
        
        super().__init__(data_path, use_memory_mapping, cache_size, prefetch_size)
        
        # Load data
        self._load_data()
        
        logger.info(f"CSVDataset initialized: {len(self.data_paths)} files, {len(self)} samples")
    
    def _load_data(self):
        """Load CSV data with optimizations."""
        if self.use_dask and len(self.data_paths) > 1:
            self._load_with_dask()
        else:
            self._load_with_pandas()
        
        # Extract feature and target information
        if self.combined_df is not None:
            self._setup_columns()
    
    def _load_with_dask(self):
        """Load data using Dask for out-of-core processing."""
        try:
            file_paths = [str(path) for path in self.data_paths]
            
            # Load with Dask
            self.dask_df = dd.read_csv(
                file_paths,
                dtype=self.dtype_dict,
                **self.pandas_kwargs
            )
            
            # Convert to pandas for easier indexing (if small enough)
            try:
                # Check size estimate
                memory_usage = self.dask_df.memory_usage(deep=True).sum().compute()
                
                if memory_usage < 1024**3:  # Less than 1GB
                    self.combined_df = self.dask_df.compute()
                    logger.info("Converted Dask DataFrame to Pandas (fits in memory)")
                else:
                    logger.info("Keeping as Dask DataFrame (too large for memory)")
            
            except Exception as e:
                logger.warning(f"Failed to convert Dask to Pandas: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to load with Dask: {e}. Falling back to Pandas.")
            self._load_with_pandas()
    
    def _load_with_pandas(self):
        """Load data using Pandas with chunked reading."""
        total_rows = 0
        
        for i, path in enumerate(self.data_paths):
            try:
                # Read CSV with chunking for large files
                if path.stat().st_size > 100 * 1024**2:  # > 100MB
                    chunks = []
                    for chunk in pd.read_csv(
                        path,
                        chunksize=self.chunk_size,
                        dtype=self.dtype_dict,
                        **self.pandas_kwargs
                    ):
                        chunks.append(chunk)
                    
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(
                        path,
                        dtype=self.dtype_dict,
                        **self.pandas_kwargs
                    )
                
                # Track file boundaries
                start_idx = total_rows
                end_idx = total_rows + len(df)
                self.file_indices.append((start_idx, end_idx))
                
                self.dataframes.append(df)
                total_rows += len(df)
                
                logger.debug(f"Loaded {path}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                raise
        
        # Combine all dataframes if manageable
        if total_rows < 10**6:  # Less than 1M rows
            self.combined_df = pd.concat(self.dataframes, ignore_index=True)
            logger.info(f"Combined {len(self.dataframes)} DataFrames into single DataFrame")
    
    def _setup_columns(self):
        """Setup feature and target columns."""
        df = self.combined_df if self.combined_df is not None else self.dataframes[0]
        
        if self.feature_columns is None:
            # Use all columns except target as features
            if self.target_column and self.target_column in df.columns:
                self.feature_columns = [col for col in df.columns if col != self.target_column]
            else:
                self.feature_columns = list(df.columns)
        
        # Validate columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        
        if self.target_column and self.target_column not in df.columns:
            raise ValueError(f"Target column not found: {self.target_column}")
        
        logger.info(f"Features: {len(self.feature_columns)} columns, Target: {self.target_column}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._length is not None:
            return self._length
        
        if self.combined_df is not None:
            self._length = len(self.combined_df)
        elif self.dask_df is not None:
            self._length = len(self.dask_df)
        else:
            self._length = sum(len(df) for df in self.dataframes)
        
        return self._length
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # Check cache first
        cached_item = self._get_from_cache(index)
        if cached_item is not None:
            return cached_item
        
        # Get the row
        try:
            if self.combined_df is not None:
                row = self.combined_df.iloc[index]
            elif self.dask_df is not None:
                # Less efficient for random access
                row = self.dask_df.iloc[index].compute()
            else:
                # Find the correct dataframe
                row = self._get_row_from_multiple_dfs(index)
            
            # Convert to tensors
            item = self._row_to_tensors(row)
            
            # Apply transform if provided
            if self.transform:
                item = self.transform(item)
            
            # Cache the item
            self._put_in_cache(index, item)
            
            return item
            
        except Exception as e:
            logger.error(f"Failed to get item {index}: {e}")
            raise
    
    def _get_row_from_multiple_dfs(self, index: int) -> pd.Series:
        """Get row from multiple dataframes using file indices."""
        for i, (start_idx, end_idx) in enumerate(self.file_indices):
            if start_idx <= index < end_idx:
                local_index = index - start_idx
                return self.dataframes[i].iloc[local_index]
        
        raise IndexError(f"Index {index} not found in any dataframe")
    
    def _row_to_tensors(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Convert a pandas row to PyTorch tensors."""
        item = {}
        
        # Features
        if self.feature_columns:
            features = row[self.feature_columns].values
            
            # Handle mixed data types
            feature_tensor = self._convert_to_tensor(features)
            item['features'] = feature_tensor
        
        # Target
        if self.target_column:
            target = row[self.target_column]
            target_tensor = self._convert_to_tensor(target)
            item['target'] = target_tensor
        
        # Include all columns if no specific setup
        if not self.feature_columns and not self.target_column:
            for col, value in row.items():
                item[col] = self._convert_to_tensor(value)
        
        return item
    
    def _convert_to_tensor(self, value: Union[np.ndarray, Any]) -> torch.Tensor:
        """Convert value to PyTorch tensor with appropriate dtype."""
        if isinstance(value, (list, np.ndarray)):
            # Handle arrays
            if len(value) == 0:
                return torch.tensor([])
            
            # Try to infer numeric type
            try:
                if all(isinstance(v, (int, np.integer)) for v in value):
                    return torch.tensor(value, dtype=torch.long)
                elif all(isinstance(v, (float, np.floating)) for v in value):
                    return torch.tensor(value, dtype=torch.float32)
                else:
                    # Mixed or string data - convert to float if possible
                    numeric_values = []
                    for v in value:
                        try:
                            numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            # Can't convert to numeric - return as is
                            return torch.tensor([hash(str(v)) % 2**31 for v in value], dtype=torch.long)
                    return torch.tensor(numeric_values, dtype=torch.float32)
            
            except Exception:
                return torch.tensor([0.0], dtype=torch.float32)  # Fallback
        
        else:
            # Handle scalar values
            try:
                if isinstance(value, (int, np.integer)):
                    return torch.tensor(value, dtype=torch.long)
                elif isinstance(value, (float, np.floating)):
                    return torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, str):
                    # Convert string to hash
                    return torch.tensor(hash(value) % 2**31, dtype=torch.long)
                else:
                    # Try to convert to float
                    return torch.tensor(float(value), dtype=torch.float32)
            
            except (ValueError, TypeError):
                return torch.tensor(0.0, dtype=torch.float32)  # Fallback
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get information about columns and data types."""
        df = self.combined_df if self.combined_df is not None else self.dataframes[0]
        
        info = {
            'total_columns': len(df.columns),
            'feature_columns': len(self.feature_columns) if self.feature_columns else 0,
            'target_column': self.target_column,
            'column_dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # Add basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            info['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        return info


class CSVLoader(BaseOptimizedLoader):
    """
    GPU-aware optimized CSV data loader.
    
    Features:
    - Memory-mapped I/O for large CSV files
    - Chunked loading for out-of-core processing
    - Dask integration for parallel processing
    - Automatic data type inference and optimization
    - Intelligent caching and prefetching
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        enable_zero_copy: bool = True,
        chunk_size: int = 10000,
        use_dask: bool = True,
        dtype_dict: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        **pandas_kwargs
    ):
        """
        Initialize CSV loader.
        
        Args:
            data_path: Path(s) to CSV file(s)
            target_column: Name of target/label column
            feature_columns: List of feature column names (None = all except target)
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Use pinned memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            prefetch_factor: Number of batches to prefetch per worker
            use_memory_mapping: Use memory-mapped I/O
            cache_size: Size of LRU cache
            enable_zero_copy: Enable zero-copy operations
            chunk_size: Chunk size for reading large files
            use_dask: Use Dask for parallel processing
            dtype_dict: Dictionary of column names to data types
            transform: Optional transform function
            **pandas_kwargs: Additional arguments for pandas.read_csv
        """
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.dtype_dict = dtype_dict
        self.transform = transform
        self.pandas_kwargs = pandas_kwargs
        
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            use_memory_mapping=use_memory_mapping,
            cache_size=cache_size,
            enable_zero_copy=enable_zero_copy
        )
        
        logger.info(f"CSVLoader initialized for {len(self.data_paths) if isinstance(data_path, list) else 1} files")
    
    def _create_dataset(self) -> CSVDataset:
        """Create the optimized CSV dataset."""
        return CSVDataset(
            data_path=self.data_path,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            transform=self.transform,
            use_memory_mapping=self.use_memory_mapping,
            cache_size=self.cache_size,
            prefetch_size=self.prefetch_factor * self.batch_size,
            chunk_size=self.chunk_size,
            use_dask=self.use_dask,
            dtype_dict=self.dtype_dict,
            **self.pandas_kwargs
        )
    
    def analyze_data_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of data for optimization insights."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        analysis = {}
        
        # Get column information
        column_info = self.dataset.get_column_info()
        analysis['column_info'] = column_info
        
        # Memory usage analysis
        analysis['memory_analysis'] = {
            'total_memory_mb': column_info['memory_usage_mb'],
            'memory_per_sample_bytes': (column_info['memory_usage_mb'] * 1024**2) / len(self.dataset),
            'cache_efficiency': min(1.0, self.cache_size / len(self.dataset))
        }
        
        # Batch size recommendations
        sample_memory_mb = analysis['memory_analysis']['memory_per_sample_bytes'] / (1024**2)
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            recommended_batch_size = int(gpu_memory_mb * 0.5 / max(sample_memory_mb, 0.001))  # Use 50% of GPU memory
            analysis['batch_size_recommendation'] = {
                'current': self.batch_size,
                'recommended': max(1, min(recommended_batch_size, 1024)),
                'reasoning': f"Based on {sample_memory_mb:.3f}MB per sample and {gpu_memory_mb:.0f}MB GPU memory"
            }
        
        return analysis
    
    def optimize_dtypes(self) -> Dict[str, Any]:
        """Optimize data types to reduce memory usage."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        df = self.dataset.combined_df
        if df is None:
            return {'error': 'No DataFrame available for optimization'}
        
        original_memory = df.memory_usage(deep=True).sum()
        optimizations = {}
        
        for col in df.columns:
            original_dtype = df[col].dtype
            
            if original_dtype == 'object':
                # Try to convert strings to categories
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
                    optimizations[col] = f'object -> category (unique ratio: {unique_ratio:.3f})'
            
            elif 'int' in str(original_dtype):
                # Downcast integers
                new_dtype = pd.api.types.pandas_dtype(original_dtype)
                min_val, max_val = df[col].min(), df[col].max()
                
                if min_val >= 0:  # Unsigned
                    if max_val < 255:
                        df[col] = df[col].astype('uint8')
                        optimizations[col] = f'{original_dtype} -> uint8'
                    elif max_val < 65535:
                        df[col] = df[col].astype('uint16')
                        optimizations[col] = f'{original_dtype} -> uint16'
                else:  # Signed
                    if -128 <= min_val and max_val < 127:
                        df[col] = df[col].astype('int8')
                        optimizations[col] = f'{original_dtype} -> int8'
                    elif -32768 <= min_val and max_val < 32767:
                        df[col] = df[col].astype('int16')
                        optimizations[col] = f'{original_dtype} -> int16'
            
            elif 'float' in str(original_dtype):
                # Downcast floats
                if original_dtype == 'float64':
                    # Check if we can use float32 without significant precision loss
                    df_float32 = df[col].astype('float32')
                    if np.allclose(df[col], df_float32, rtol=1e-6, equal_nan=True):
                        df[col] = df_float32
                        optimizations[col] = 'float64 -> float32'
        
        new_memory = df.memory_usage(deep=True).sum()
        memory_saved = original_memory - new_memory
        
        return {
            'optimizations_applied': optimizations,
            'memory_saved_mb': memory_saved / (1024**2),
            'memory_reduction_percent': (memory_saved / original_memory) * 100,
            'original_memory_mb': original_memory / (1024**2),
            'new_memory_mb': new_memory / (1024**2)
        }
    
    def create_balanced_sampler(self, target_column: Optional[str] = None):
        """Create a balanced sampler for imbalanced datasets."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        target_col = target_column or self.target_column
        if not target_col:
            raise ValueError("No target column specified for balanced sampling")
        
        df = self.dataset.combined_df
        if df is None:
            raise ValueError("No DataFrame available for balanced sampling")
        
        from torch.utils.data import WeightedRandomSampler
        
        # Calculate class weights
        class_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        # Create weights (inverse frequency)
        class_weights = {}
        for class_label, count in class_counts.items():
            class_weights[class_label] = total_samples / count
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in df[target_col]]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler, {
            'class_counts': class_counts.to_dict(),
            'class_weights': class_weights,
            'total_samples': total_samples
        }
    
    def benchmark_loading_performance(self, num_batches: int = 50) -> Dict[str, float]:
        """Benchmark CSV loading performance."""
        import time
        
        times = []
        memory_usage = []
        
        # Warmup
        for _ in range(5):
            batch = next(iter(self))
        
        # Benchmark
        for i, batch in enumerate(self):
            if i >= num_batches:
                break
            
            start_time = time.time()
            
            # Simulate processing
            if isinstance(batch, dict):
                for key, tensor in batch.items():
                    if torch.is_tensor(tensor):
                        _ = tensor.mean()  # Simple operation
            
            batch_time = time.time() - start_time
            times.append(batch_time)
            
            # Track memory
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / (1024**2))
        
        return {
            'avg_batch_time_sec': np.mean(times),
            'std_batch_time_sec': np.std(times),
            'min_batch_time_sec': np.min(times),
            'max_batch_time_sec': np.max(times),
            'batches_per_sec': 1.0 / np.mean(times) if times else 0,
            'avg_gpu_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'peak_gpu_memory_mb': np.max(memory_usage) if memory_usage else 0
        }
    
    @staticmethod
    def suggest_dtype_optimizations(file_path: Union[str, Path]) -> Dict[str, str]:
        """Analyze a CSV file and suggest optimal data types."""
        # Read a sample to infer types
        sample_df = pd.read_csv(file_path, nrows=1000)
        suggestions = {}
        
        for col in sample_df.columns:
            dtype = sample_df[col].dtype
            
            if dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(sample_df[col])
                    suggestions[col] = 'float32'  # Default numeric type
                except ValueError:
                    # Check uniqueness for categorical
                    unique_ratio = sample_df[col].nunique() / len(sample_df)
                    if unique_ratio < 0.5:
                        suggestions[col] = 'category'
                    else:
                        suggestions[col] = 'object'  # Keep as string
            
            elif 'int' in str(dtype):
                # Suggest smaller int types based on range
                min_val, max_val = sample_df[col].min(), sample_df[col].max()
                if min_val >= 0:
                    if max_val < 255:
                        suggestions[col] = 'uint8'
                    elif max_val < 65535:
                        suggestions[col] = 'uint16'
                    else:
                        suggestions[col] = 'uint32'
                else:
                    if -128 <= min_val and max_val < 127:
                        suggestions[col] = 'int8'
                    elif -32768 <= min_val and max_val < 32767:
                        suggestions[col] = 'int16'
                    else:
                        suggestions[col] = 'int32'
            
            elif dtype == 'float64':
                suggestions[col] = 'float32'  # Usually sufficient precision
            
            else:
                suggestions[col] = str(dtype)  # Keep as is
        
        return suggestions 