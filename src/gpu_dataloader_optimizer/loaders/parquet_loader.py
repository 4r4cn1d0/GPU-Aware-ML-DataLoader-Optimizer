"""
Optimized Parquet data loader with GPU-aware optimizations.
"""

from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None

from .base_loader import BaseOptimizedLoader, OptimizedDataset


logger = logging.getLogger(__name__)


class ParquetDataset(OptimizedDataset):
    """Optimized Parquet dataset with memory mapping and columnar access."""
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        use_memory_mapping: bool = True,
        cache_size: int = 1000,
        prefetch_size: int = 100,
        use_dask: bool = True,
        columns: Optional[List[str]] = None,
        filters: Optional[List[List[tuple]]] = None,
        **parquet_kwargs
    ):
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for Parquet loading. Install with: pip install pyarrow")
        
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.use_dask = use_dask and DASK_AVAILABLE
        self.columns = columns
        self.filters = filters
        self.parquet_kwargs = parquet_kwargs
        
        # Parquet-specific storage
        self.parquet_files: List[pq.ParquetFile] = []
        self.table_metadata: List[pq.ParquetFile] = []
        self.dask_df: Optional[dd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
        
        # Row group information for efficient access
        self.row_group_info: List[Dict[str, Any]] = []
        self._length: Optional[int] = None
        
        super().__init__(data_path, use_memory_mapping, cache_size, prefetch_size)
        
        # Load parquet metadata and data
        self._load_parquet_metadata()
        self._load_data()
        
        logger.info(f"ParquetDataset initialized: {len(self.data_paths)} files, {len(self)} samples")
    
    def _load_parquet_metadata(self):
        """Load Parquet file metadata for efficient access."""
        for path in self.data_paths:
            try:
                parquet_file = pq.ParquetFile(str(path))
                self.parquet_files.append(parquet_file)
                
                # Extract row group information
                metadata = parquet_file.metadata
                schema = parquet_file.schema
                
                file_info = {
                    'path': path,
                    'num_rows': metadata.num_rows,
                    'num_row_groups': metadata.num_row_groups,
                    'schema': schema,
                    'columns': [field.name for field in schema],
                    'row_groups': []
                }
                
                # Get row group details
                for i in range(metadata.num_row_groups):
                    rg = metadata.row_group(i)
                    file_info['row_groups'].append({
                        'index': i,
                        'num_rows': rg.num_rows,
                        'total_byte_size': rg.total_byte_size
                    })
                
                self.row_group_info.append(file_info)
                
                logger.debug(f"Loaded metadata for {path}: {metadata.num_rows} rows, "
                           f"{metadata.num_row_groups} row groups")
                
            except Exception as e:
                logger.error(f"Failed to load metadata for {path}: {e}")
                raise
    
    def _load_data(self):
        """Load Parquet data with optimizations."""
        if self.use_dask and len(self.data_paths) > 1:
            self._load_with_dask()
        else:
            self._load_with_pyarrow()
        
        # Setup column information
        if self.combined_df is not None or self.dask_df is not None:
            self._setup_columns()
    
    def _load_with_dask(self):
        """Load data using Dask for out-of-core processing."""
        try:
            file_paths = [str(path) for path in self.data_paths]
            
            # Load with Dask
            self.dask_df = dd.read_parquet(
                file_paths,
                columns=self.columns,
                filters=self.filters,
                **self.parquet_kwargs
            )
            
            # Convert to pandas if small enough
            try:
                # Estimate memory usage
                nrows = len(self.dask_df)
                ncols = len(self.dask_df.columns)
                estimated_memory = nrows * ncols * 8  # Rough estimate (8 bytes per value)
                
                if estimated_memory < 1024**3:  # Less than 1GB
                    self.combined_df = self.dask_df.compute()
                    logger.info("Converted Dask DataFrame to Pandas (fits in memory)")
                else:
                    logger.info("Keeping as Dask DataFrame (too large for memory)")
            
            except Exception as e:
                logger.warning(f"Failed to convert Dask to Pandas: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to load with Dask: {e}. Falling back to PyArrow.")
            self._load_with_pyarrow()
    
    def _load_with_pyarrow(self):
        """Load data using PyArrow with efficient columnar access."""
        try:
            if len(self.data_paths) == 1:
                # Single file
                table = pq.read_table(
                    str(self.data_paths[0]),
                    columns=self.columns,
                    filters=self.filters,
                    **self.parquet_kwargs
                )
                self.combined_df = table.to_pandas()
            
            else:
                # Multiple files
                tables = []
                for path in self.data_paths:
                    table = pq.read_table(
                        str(path),
                        columns=self.columns,
                        filters=self.filters,
                        **self.parquet_kwargs
                    )
                    tables.append(table)
                
                # Concatenate tables
                combined_table = pa.concat_tables(tables)
                self.combined_df = combined_table.to_pandas()
            
            logger.info(f"Loaded {len(self.combined_df)} rows using PyArrow")
            
        except Exception as e:
            logger.error(f"Failed to load with PyArrow: {e}")
            raise
    
    def _setup_columns(self):
        """Setup feature and target columns."""
        df = self.combined_df if self.combined_df is not None else self.dask_df
        
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
            # Sum from metadata
            self._length = sum(info['num_rows'] for info in self.row_group_info)
        
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
                # Direct row group access (most efficient for large files)
                row = self._get_row_from_row_groups(index)
            
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
    
    def _get_row_from_row_groups(self, index: int) -> pd.Series:
        """Get row directly from row groups (most efficient for large files)."""
        current_offset = 0
        
        for file_info in self.row_group_info:
            file_end = current_offset + file_info['num_rows']
            
            if current_offset <= index < file_end:
                # Found the file
                local_index = index - current_offset
                
                # Find the row group
                rg_offset = 0
                for rg_info in file_info['row_groups']:
                    rg_end = rg_offset + rg_info['num_rows']
                    
                    if rg_offset <= local_index < rg_end:
                        # Found the row group
                        rg_local_index = local_index - rg_offset
                        
                        # Read the specific row group
                        table = pq.read_table(
                            str(file_info['path']),
                            columns=self.columns,
                            filters=self.filters,
                            **self.parquet_kwargs
                        )
                        
                        df = table.to_pandas()
                        return df.iloc[rg_local_index]
                    
                    rg_offset = rg_end
            
            current_offset = file_end
        
        raise IndexError(f"Index {index} not found in any row group")
    
    def _row_to_tensors(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Convert a pandas row to PyTorch tensors."""
        item = {}
        
        # Features
        if self.feature_columns:
            features = row[self.feature_columns].values
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
            if len(value) == 0:
                return torch.tensor([])
            
            # Handle different numpy dtypes efficiently
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.integer):
                    return torch.from_numpy(value.astype(np.int64))
                elif np.issubdtype(value.dtype, np.floating):
                    return torch.from_numpy(value.astype(np.float32))
                elif value.dtype == np.bool_:
                    return torch.from_numpy(value.astype(np.int64))
            
            # Fallback to generic conversion
            try:
                if all(isinstance(v, (int, np.integer)) for v in value):
                    return torch.tensor(value, dtype=torch.long)
                elif all(isinstance(v, (float, np.floating)) for v in value):
                    return torch.tensor(value, dtype=torch.float32)
                else:
                    # Try to convert to numeric
                    numeric_values = []
                    for v in value:
                        try:
                            numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            # Hash strings/objects
                            numeric_values.append(hash(str(v)) % 2**31)
                    return torch.tensor(numeric_values, dtype=torch.float32)
            
            except Exception:
                return torch.tensor([0.0], dtype=torch.float32)
        
        else:
            # Handle scalar values
            try:
                if isinstance(value, (int, np.integer)):
                    return torch.tensor(value, dtype=torch.long)
                elif isinstance(value, (float, np.floating)):
                    return torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, (bool, np.bool_)):
                    return torch.tensor(int(value), dtype=torch.long)
                elif isinstance(value, str):
                    return torch.tensor(hash(value) % 2**31, dtype=torch.long)
                else:
                    return torch.tensor(float(value), dtype=torch.float32)
            
            except (ValueError, TypeError):
                return torch.tensor(0.0, dtype=torch.float32)
    
    def get_parquet_info(self) -> Dict[str, Any]:
        """Get detailed information about Parquet files."""
        info = {
            'total_files': len(self.data_paths),
            'total_rows': sum(info['num_rows'] for info in self.row_group_info),
            'total_row_groups': sum(info['num_row_groups'] for info in self.row_group_info),
            'files': []
        }
        
        for file_info in self.row_group_info:
            file_details = {
                'path': str(file_info['path']),
                'num_rows': file_info['num_rows'],
                'num_row_groups': file_info['num_row_groups'],
                'columns': file_info['columns'],
                'total_size_bytes': sum(rg['total_byte_size'] for rg in file_info['row_groups']),
                'avg_row_group_size_bytes': np.mean([rg['total_byte_size'] for rg in file_info['row_groups']])
            }
            info['files'].append(file_details)
        
        return info


class ParquetLoader(BaseOptimizedLoader):
    """
    GPU-aware optimized Parquet data loader.
    
    Features:
    - Memory-mapped I/O for large Parquet files
    - Efficient columnar access using PyArrow
    - Row group-level optimization
    - Dask integration for parallel processing
    - Predicate pushdown filtering
    - Zero-copy operations where possible
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
        use_dask: bool = True,
        columns: Optional[List[str]] = None,
        filters: Optional[List[List[tuple]]] = None,
        transform: Optional[Callable] = None,
        **parquet_kwargs
    ):
        """
        Initialize Parquet loader.
        
        Args:
            data_path: Path(s) to Parquet file(s)
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
            use_dask: Use Dask for parallel processing
            columns: Specific columns to load (for column pruning)
            filters: PyArrow filters for predicate pushdown
            transform: Optional transform function
            **parquet_kwargs: Additional arguments for PyArrow
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for Parquet loading. Install with: pip install pyarrow")
        
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.use_dask = use_dask
        self.columns = columns
        self.filters = filters
        self.transform = transform
        self.parquet_kwargs = parquet_kwargs
        
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
        
        logger.info(f"ParquetLoader initialized for {len(self.data_paths) if isinstance(data_path, list) else 1} files")
    
    def _create_dataset(self) -> ParquetDataset:
        """Create the optimized Parquet dataset."""
        return ParquetDataset(
            data_path=self.data_path,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            transform=self.transform,
            use_memory_mapping=self.use_memory_mapping,
            cache_size=self.cache_size,
            prefetch_size=self.prefetch_factor * self.batch_size,
            use_dask=self.use_dask,
            columns=self.columns,
            filters=self.filters,
            **self.parquet_kwargs
        )
    
    def analyze_column_statistics(self) -> Dict[str, Any]:
        """Analyze column statistics for optimization insights."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        analysis = {}
        
        # Get Parquet metadata
        parquet_info = self.dataset.get_parquet_info()
        analysis['parquet_info'] = parquet_info
        
        # Analyze schema and data types
        if self.dataset.row_group_info:
            first_file = self.dataset.row_group_info[0]
            schema = first_file['schema']
            
            column_analysis = {}
            for field in schema:
                column_analysis[field.name] = {
                    'type': str(field.type),
                    'logical_type': str(field.logical_type) if hasattr(field, 'logical_type') else None,
                    'physical_type': str(field.physical_type) if hasattr(field, 'physical_type') else None
                }
            
            analysis['column_types'] = column_analysis
        
        # Memory usage analysis
        if self.dataset.combined_df is not None:
            df = self.dataset.combined_df
            analysis['memory_analysis'] = {
                'total_memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'memory_per_sample_bytes': df.memory_usage(deep=True).sum() / len(df),
                'column_memory_usage': {col: df[col].memory_usage(deep=True) / (1024**2) 
                                      for col in df.columns}
            }
        
        return analysis
    
    def optimize_column_selection(self) -> Dict[str, Any]:
        """Suggest optimal column selection for memory efficiency."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        if not self.dataset.combined_df is not None:
            return {'error': 'No DataFrame available for column optimization'}
        
        df = self.dataset.combined_df
        optimization = {
            'current_columns': len(df.columns),
            'suggestions': [],
            'memory_impact': {}
        }
        
        # Analyze column importance (basic heuristics)
        for col in df.columns:
            col_memory = df[col].memory_usage(deep=True) / (1024**2)
            unique_ratio = df[col].nunique() / len(df)
            null_ratio = df[col].isnull().sum() / len(df)
            
            suggestion = {
                'column': col,
                'memory_mb': col_memory,
                'unique_ratio': unique_ratio,
                'null_ratio': null_ratio,
                'recommendation': 'keep'
            }
            
            # Suggest dropping columns with high null ratio and low uniqueness
            if null_ratio > 0.8:
                suggestion['recommendation'] = 'consider_dropping'
                suggestion['reason'] = f'High null ratio: {null_ratio:.2%}'
            elif unique_ratio < 0.01 and col != self.target_column:
                suggestion['recommendation'] = 'consider_dropping'
                suggestion['reason'] = f'Low uniqueness: {unique_ratio:.2%}'
            elif col_memory > 50 and unique_ratio > 0.9:
                suggestion['recommendation'] = 'consider_feature_engineering'
                suggestion['reason'] = f'High memory, high cardinality: {col_memory:.1f}MB, {unique_ratio:.2%} unique'
            
            optimization['suggestions'].append(suggestion)
        
        # Calculate potential memory savings
        droppable_columns = [s for s in optimization['suggestions'] 
                           if s['recommendation'] == 'consider_dropping']
        potential_savings = sum(s['memory_mb'] for s in droppable_columns)
        
        optimization['memory_impact'] = {
            'current_memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'potential_savings_mb': potential_savings,
            'savings_percent': (potential_savings / (df.memory_usage(deep=True).sum() / (1024**2))) * 100
        }
        
        return optimization
    
    def create_filtered_loader(self, filters: List[List[tuple]]) -> 'ParquetLoader':
        """Create a new loader with additional filters applied."""
        return ParquetLoader(
            data_path=self.data_path,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            use_memory_mapping=self.use_memory_mapping,
            cache_size=self.cache_size,
            enable_zero_copy=self.enable_zero_copy,
            use_dask=self.use_dask,
            columns=self.columns,
            filters=filters,
            transform=self.transform,
            **self.parquet_kwargs
        )
    
    def benchmark_row_group_access(self) -> Dict[str, Any]:
        """Benchmark different access patterns for row groups."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        
        import time
        
        benchmarks = {}
        
        # Sequential access benchmark
        start_time = time.time()
        for i in range(min(1000, len(self.dataset))):
            _ = self.dataset[i]
        sequential_time = time.time() - start_time
        
        benchmarks['sequential_access'] = {
            'time_sec': sequential_time,
            'samples_per_sec': 1000 / sequential_time if sequential_time > 0 else 0
        }
        
        # Random access benchmark
        import random
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(100)]
        
        start_time = time.time()
        for idx in indices:
            _ = self.dataset[idx]
        random_time = time.time() - start_time
        
        benchmarks['random_access'] = {
            'time_sec': random_time,
            'samples_per_sec': 100 / random_time if random_time > 0 else 0
        }
        
        # Batch access benchmark
        start_time = time.time()
        for batch in self:
            break  # Just time one batch
        batch_time = time.time() - start_time
        
        benchmarks['batch_access'] = {
            'time_sec': batch_time,
            'batch_size': self.batch_size,
            'samples_per_sec': self.batch_size / batch_time if batch_time > 0 else 0
        }
        
        return benchmarks
    
    @staticmethod
    def analyze_parquet_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze a Parquet file and provide optimization recommendations."""
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required")
        
        file_path = Path(file_path)
        
        try:
            parquet_file = pq.ParquetFile(str(file_path))
            metadata = parquet_file.metadata
            schema = parquet_file.schema
            
            analysis = {
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024**2),
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'num_row_groups': metadata.num_row_groups,
                'created_by': metadata.created_by,
                'schema': {field.name: str(field.type) for field in schema},
                'row_groups': [],
                'recommendations': []
            }
            
            # Analyze row groups
            for i in range(metadata.num_row_groups):
                rg = metadata.row_group(i)
                rg_info = {
                    'index': i,
                    'num_rows': rg.num_rows,
                    'total_byte_size': rg.total_byte_size,
                    'compressed_size': rg.total_compressed_size if hasattr(rg, 'total_compressed_size') else 0
                }
                analysis['row_groups'].append(rg_info)
            
            # Generate recommendations
            avg_rg_size = np.mean([rg['total_byte_size'] for rg in analysis['row_groups']])
            
            if avg_rg_size < 64 * 1024**2:  # Less than 64MB
                analysis['recommendations'].append(
                    "Consider using larger row groups (64-256MB) for better I/O efficiency"
                )
            
            if metadata.num_row_groups > 100:
                analysis['recommendations'].append(
                    "Large number of row groups may impact performance. Consider consolidating."
                )
            
            if analysis['file_size_mb'] > 1000:  # > 1GB
                analysis['recommendations'].append(
                    "Large file detected. Consider partitioning for better parallel access."
                )
            
            return analysis
            
        except Exception as e:
            return {'error': f"Failed to analyze file: {e}"}
    
    @staticmethod
    def suggest_filters_from_data(df: pd.DataFrame, target_column: str = None) -> List[List[tuple]]:
        """Suggest efficient filters based on data distribution."""
        suggestions = []
        
        for col in df.columns:
            if col == target_column:
                continue
            
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            
            # Suggest filters for low-cardinality columns
            if unique_count < 100 and dtype in ['object', 'category']:
                value_counts = df[col].value_counts()
                top_values = value_counts.head(5).index.tolist()
                
                for value in top_values:
                    suggestions.append([[(col, '==', value)]])
            
            # Suggest range filters for numeric columns
            elif np.issubdtype(dtype, np.number):
                q25, q75 = df[col].quantile([0.25, 0.75])
                suggestions.append([[(col, '>=', q25), (col, '<=', q75)]])
        
        return suggestions[:10]  # Return top 10 suggestions 