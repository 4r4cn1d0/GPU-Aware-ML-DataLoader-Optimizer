"""
Configuration utilities and data models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import yaml
import json

from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class ProfilerConfig(BaseModel):
    """Configuration for the DataLoaderProfiler."""
    
    # Profiling parameters
    default_iterations: int = Field(default=100, ge=10, le=10000)
    warmup_iterations: int = Field(default=10, ge=0, le=100)
    sample_interval: float = Field(default=0.1, ge=0.01, le=1.0)
    
    # Resource monitoring
    monitor_gpu: bool = Field(default=True)
    monitor_system: bool = Field(default=True)
    gpu_sample_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    system_sample_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    
    # Optimization settings
    enable_profiling_optimizations: bool = Field(default=True)
    parallel_profiling: bool = Field(default=True)
    max_parallel_configs: int = Field(default=4, ge=1, le=16)
    
    # Output settings
    save_detailed_logs: bool = Field(default=True)
    export_format: str = Field(default="json", regex="^(json|csv|pickle)$")
    
    class Config:
        extra = "forbid"  # Don't allow extra fields


class OptimizerConfig(BaseModel):
    """Configuration for the DataLoaderOptimizer."""
    
    # Machine learning parameters
    min_training_samples: int = Field(default=50, ge=10, le=10000)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42, ge=0)
    
    # Model parameters
    rf_n_estimators: int = Field(default=100, ge=10, le=1000)
    rf_max_depth: Optional[int] = Field(default=None)
    rf_min_samples_split: int = Field(default=2, ge=2)
    
    # Optimization parameters
    optimization_targets: List[str] = Field(
        default=["throughput", "memory", "stability"], 
        min_items=1
    )
    constraint_tolerance: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Feature engineering
    enable_feature_engineering: bool = Field(default=True)
    feature_selection_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    
    @validator('optimization_targets')
    def validate_targets(cls, v):
        valid_targets = {"throughput", "memory", "stability", "speed", "balanced"}
        if not all(target in valid_targets for target in v):
            raise ValueError(f"Invalid optimization targets. Must be from: {valid_targets}")
        return v
    
    class Config:
        extra = "forbid"


class LoaderConfig(BaseModel):
    """Configuration for data loaders."""
    
    # Basic loader settings
    batch_size: int = Field(default=32, ge=1, le=2048)
    num_workers: int = Field(default=4, ge=0, le=32)
    pin_memory: bool = Field(default=True)
    drop_last: bool = Field(default=False)
    prefetch_factor: int = Field(default=2, ge=1, le=10)
    
    # Optimization settings
    use_memory_mapping: bool = Field(default=True)
    cache_size: int = Field(default=1000, ge=0, le=100000)
    enable_zero_copy: bool = Field(default=True)
    enable_prefetch: bool = Field(default=True)
    
    # GPU settings
    gpu_id: int = Field(default=0, ge=0)
    memory_limit_mb: Optional[float] = Field(default=None, ge=100.0)
    
    # Format-specific settings
    csv_chunk_size: int = Field(default=10000, ge=1000, le=1000000)
    csv_use_dask: bool = Field(default=True)
    
    tfrecord_compression: str = Field(default="AUTO", regex="^(AUTO|GZIP|ZLIB|)$")
    tfrecord_parallel_reads: int = Field(default=4, ge=1, le=16)
    
    parquet_use_dask: bool = Field(default=True)
    parquet_row_group_size: Optional[int] = Field(default=None, ge=1000)
    
    class Config:
        extra = "forbid"


class BenchmarkConfig(BaseModel):
    """Configuration for ML model benchmarking."""
    
    # General benchmark settings
    enabled_models: List[str] = Field(
        default=["resnet18", "vit_base", "bert_base"],
        min_items=1
    )
    benchmark_iterations: int = Field(default=50, ge=10, le=1000)
    warmup_iterations: int = Field(default=5, ge=0, le=50)
    
    # Model-specific settings
    vision_input_size: tuple = Field(default=(224, 224, 3))
    text_sequence_length: int = Field(default=512, ge=64, le=2048)
    
    # Hardware settings
    use_mixed_precision: bool = Field(default=True)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32)
    
    # Metrics to collect
    collect_flops: bool = Field(default=True)
    collect_memory_breakdown: bool = Field(default=True)
    collect_timing_breakdown: bool = Field(default=True)
    
    @validator('enabled_models')
    def validate_models(cls, v):
        valid_models = {
            "resnet18", "resnet50", "resnet101",
            "vit_base", "vit_large", 
            "bert_base", "bert_large", "gpt2", "t5_base"
        }
        if not all(model in valid_models for model in v):
            raise ValueError(f"Invalid models. Must be from: {valid_models}")
        return v
    
    class Config:
        extra = "forbid"


class SystemConfig(BaseModel):
    """Overall system configuration."""
    
    # Global settings
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    output_dir: Path = Field(default=Path("./gpu_dataloader_results"))
    temp_dir: Optional[Path] = Field(default=None)
    
    # Component configurations
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    
    # Resource limits
    max_memory_usage_gb: Optional[float] = Field(default=None, ge=1.0)
    max_gpu_memory_usage_percent: float = Field(default=0.8, ge=0.1, le=1.0)
    max_cpu_usage_percent: float = Field(default=0.8, ge=0.1, le=1.0)
    
    # Advanced settings
    enable_distributed: bool = Field(default=False)
    distributed_backend: str = Field(default="nccl", regex="^(nccl|gloo|mpi)$")
    
    @validator('output_dir', 'temp_dir')
    def validate_paths(cls, v):
        if v is not None:
            v = Path(v)
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        extra = "forbid"


@dataclass
class ExperimentConfig:
    """Configuration for a specific profiling experiment."""
    
    experiment_name: str
    data_paths: List[Path]
    data_formats: List[str]
    batch_sizes: List[int]
    
    # Optional experiment parameters
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    models_to_benchmark: Optional[List[str]] = None
    
    # Constraints
    gpu_memory_limit_mb: Optional[float] = None
    min_throughput_samples_per_sec: Optional[float] = None
    max_latency_ms: Optional[float] = None
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate the experiment configuration and return any errors."""
        errors = []
        
        if not self.experiment_name:
            errors.append("Experiment name is required")
        
        if not self.data_paths:
            errors.append("At least one data path is required")
        
        for path in self.data_paths:
            if not path.exists():
                errors.append(f"Data path does not exist: {path}")
        
        if not self.data_formats:
            errors.append("At least one data format is required")
        
        valid_formats = {"csv", "parquet", "tfrecord"}
        for fmt in self.data_formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid data format: {fmt}. Must be one of: {valid_formats}")
        
        if not self.batch_sizes:
            errors.append("At least one batch size is required")
        
        if any(bs <= 0 for bs in self.batch_sizes):
            errors.append("All batch sizes must be positive")
        
        return errors


class ConfigManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
    
    def load_config(self, config_path: Optional[Path] = None) -> SystemConfig:
        """Load configuration from file."""
        path = config_path or self.config_path
        
        if path is None:
            logger.info("No config path provided, using default configuration")
            self.config = SystemConfig()
            return self.config
        
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using default configuration.")
            self.config = SystemConfig()
            return self.config
        
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    config_dict = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            self.config = SystemConfig(**config_dict)
            logger.info(f"Loaded configuration from: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            logger.info("Using default configuration")
            self.config = SystemConfig()
        
        return self.config
    
    def save_config(self, config: SystemConfig, save_path: Optional[Path] = None) -> Path:
        """Save configuration to file."""
        path = save_path or self.config_path
        
        if path is None:
            path = Path("gpu_dataloader_config.yaml")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.dict()
            
            with open(path, 'w') as f:
                if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    # Default to YAML
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration."""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update specific configuration values."""
        if self.config is None:
            self.config = SystemConfig()
        
        # Create a new config with updates
        config_dict = self.config.dict()
        config_dict.update(updates)
        
        self.config = SystemConfig(**config_dict)
    
    @staticmethod
    def create_default_config() -> SystemConfig:
        """Create a default configuration."""
        return SystemConfig()
    
    @staticmethod
    def create_experiment_config(
        name: str,
        data_paths: List[Union[str, Path]],
        data_formats: List[str],
        batch_sizes: List[int],
        **kwargs
    ) -> ExperimentConfig:
        """Create an experiment configuration."""
        return ExperimentConfig(
            experiment_name=name,
            data_paths=[Path(p) for p in data_paths],
            data_formats=data_formats,
            batch_sizes=batch_sizes,
            **kwargs
        )
    
    @staticmethod
    def validate_hardware_compatibility(config: SystemConfig) -> List[str]:
        """Validate that the configuration is compatible with current hardware."""
        warnings = []
        
        try:
            import torch
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                warnings.append("CUDA not available - GPU optimizations will be disabled")
            else:
                # Check GPU memory
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                if config.loader.memory_limit_mb and config.loader.memory_limit_mb > gpu_memory_mb:
                    warnings.append(f"Memory limit ({config.loader.memory_limit_mb}MB) exceeds GPU memory ({gpu_memory_mb:.0f}MB)")
            
            # Check CPU cores for num_workers
            import os
            cpu_count = os.cpu_count() or 1
            if config.loader.num_workers > cpu_count:
                warnings.append(f"num_workers ({config.loader.num_workers}) exceeds CPU count ({cpu_count})")
        
        except ImportError:
            warnings.append("PyTorch not available - functionality will be limited")
        
        return warnings


def load_config_from_env() -> SystemConfig:
    """Load configuration from environment variables."""
    import os
    
    config_dict = {}
    
    # Map environment variables to config fields
    env_mappings = {
        'GPU_DATALOADER_LOG_LEVEL': 'log_level',
        'GPU_DATALOADER_OUTPUT_DIR': 'output_dir',
        'GPU_DATALOADER_BATCH_SIZE': 'loader.batch_size',
        'GPU_DATALOADER_NUM_WORKERS': 'loader.num_workers',
        'GPU_DATALOADER_GPU_ID': 'loader.gpu_id',
        'GPU_DATALOADER_MEMORY_LIMIT': 'loader.memory_limit_mb',
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Handle nested config paths
            keys = config_path.split('.')
            current = config_dict
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Convert value to appropriate type
            final_key = keys[-1]
            if final_key in ['batch_size', 'num_workers', 'gpu_id']:
                current[final_key] = int(value)
            elif final_key in ['memory_limit_mb']:
                current[final_key] = float(value)
            else:
                current[final_key] = value
    
    return SystemConfig(**config_dict)


def create_config_template(output_path: Path):
    """Create a configuration template file."""
    config = SystemConfig()
    
    template_content = f"""# GPU-Aware ML DataLoader Optimizer Configuration
# This file contains all available configuration options with their default values

{yaml.dump(config.dict(), default_flow_style=False, indent=2)}

# Additional Notes:
# - All paths can be absolute or relative
# - Memory limits are in MB unless specified otherwise
# - Batch sizes will be automatically adjusted if they exceed GPU memory
# - Set log_level to DEBUG for verbose output during development
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    logger.info(f"Created configuration template: {output_path}")


if __name__ == "__main__":
    # Create a sample configuration template
    create_config_template(Path("config_template.yaml")) 