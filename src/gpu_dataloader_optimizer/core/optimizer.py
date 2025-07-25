"""
DataLoader optimizer that analyzes profiling results and suggests optimal configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import logging
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .profiler import ProfilingResult, ProfilingSession
from .learning_engine import LearningEngine
from ..utils.config_utils import OptimizerConfig


logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """A recommendation for optimal dataloader configuration."""
    
    # Configuration
    batch_size: int
    data_format: str
    loader_config: Dict[str, Any]
    
    # Predicted performance
    predicted_throughput: float
    predicted_memory_usage: float
    predicted_stability: float
    
    # Confidence and reasoning
    confidence_score: float
    reasoning: List[str]
    
    # Trade-off analysis
    memory_efficiency_score: float
    speed_efficiency_score: float
    stability_score: float
    overall_score: float
    
    # Constraints satisfaction
    satisfies_memory_constraint: bool
    satisfies_throughput_constraint: bool
    
    rank: int = 0


@dataclass
class OptimizationReport:
    """Complete optimization report with recommendations and analysis."""
    
    recommendations: List[OptimizationRecommendation]
    analysis: Dict[str, Any]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def get_top_recommendation(self) -> Optional[OptimizationRecommendation]:
        """Get the highest-ranked recommendation."""
        if self.recommendations:
            return min(self.recommendations, key=lambda r: r.rank)
        return None
    
    def get_recommendations_by_constraint(self, constraint: str) -> List[OptimizationRecommendation]:
        """Get recommendations that satisfy a specific constraint."""
        if constraint == 'memory':
            return [r for r in self.recommendations if r.satisfies_memory_constraint]
        elif constraint == 'throughput':
            return [r for r in self.recommendations if r.satisfies_throughput_constraint]
        else:
            return self.recommendations


class DataLoaderOptimizer:
    """
    Analyzes profiling results and suggests optimal dataloader configurations.
    
    Features:
    - Multi-objective optimization (speed, memory, stability)
    - Machine learning-based performance prediction
    - Constraint satisfaction
    - Trade-off analysis
    - Historical learning from past optimizations
    """
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.learning_engine = LearningEngine()
        
        # ML models for performance prediction
        self.throughput_model: Optional[RandomForestRegressor] = None
        self.memory_model: Optional[RandomForestRegressor] = None
        self.stability_model: Optional[RandomForestRegressor] = None
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
        # Historical data
        self.training_data: List[ProfilingResult] = []
        self.is_trained = False
        
        logger.info("DataLoaderOptimizer initialized")
    
    def add_profiling_data(self, session: ProfilingSession):
        """Add profiling data for training the optimization models."""
        self.training_data.extend(session.results)
        logger.info(f"Added {len(session.results)} profiling results to training data")
    
    def add_profiling_results(self, results: List[ProfilingResult]):
        """Add individual profiling results to training data."""
        self.training_data.extend(results)
        logger.info(f"Added {len(results)} profiling results to training data")
    
    def train_models(self, retrain: bool = False):
        """Train the ML models on available profiling data."""
        if self.is_trained and not retrain:
            logger.info("Models already trained. Use retrain=True to force retraining.")
            return
        
        if len(self.training_data) < self.config.min_training_samples:
            logger.warning(f"Insufficient training data: {len(self.training_data)} < "
                          f"{self.config.min_training_samples}")
            return
        
        logger.info(f"Training models on {len(self.training_data)} samples...")
        
        # Prepare features and targets
        features, throughput_targets, memory_targets, stability_targets = self._prepare_training_data()
        
        # Split data
        X_train, X_test, y_throughput_train, y_throughput_test = train_test_split(
            features, throughput_targets, test_size=0.2, random_state=42
        )
        _, _, y_memory_train, y_memory_test = train_test_split(
            features, memory_targets, test_size=0.2, random_state=42
        )
        _, _, y_stability_train, y_stability_test = train_test_split(
            features, stability_targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train models
        self.throughput_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.memory_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.stability_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        
        self.throughput_model.fit(X_train_scaled, y_throughput_train)
        self.memory_model.fit(X_train_scaled, y_memory_train)
        self.stability_model.fit(X_train_scaled, y_stability_train)
        
        # Evaluate models
        throughput_pred = self.throughput_model.predict(X_test_scaled)
        memory_pred = self.memory_model.predict(X_test_scaled)
        stability_pred = self.stability_model.predict(X_test_scaled)
        
        throughput_r2 = r2_score(y_throughput_test, throughput_pred)
        memory_r2 = r2_score(y_memory_test, memory_pred)
        stability_r2 = r2_score(y_stability_test, stability_pred)
        
        logger.info(f"Model performance (R²): throughput={throughput_r2:.3f}, "
                   f"memory={memory_r2:.3f}, stability={stability_r2:.3f}")
        
        self.is_trained = True
    
    def optimize(
        self,
        candidate_configs: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        optimization_target: str = 'balanced'
    ) -> OptimizationReport:
        """
        Optimize dataloader configuration given candidates and constraints.
        
        Args:
            candidate_configs: List of configuration dictionaries to evaluate
            constraints: Constraints dict with 'max_memory_mb', 'min_throughput', etc.
            optimization_target: 'speed', 'memory', 'stability', or 'balanced'
        
        Returns:
            OptimizationReport with ranked recommendations
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train_models() first.")
        
        constraints = constraints or {}
        
        logger.info(f"Optimizing {len(candidate_configs)} configurations with "
                   f"target: {optimization_target}")
        
        recommendations = []
        
        for config in candidate_configs:
            try:
                recommendation = self._evaluate_configuration(config, constraints)
                recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Failed to evaluate configuration {config}: {e}")
        
        # Rank recommendations
        recommendations = self._rank_recommendations(recommendations, optimization_target)
        
        # Generate analysis
        analysis = self._generate_analysis(recommendations, constraints)
        
        report = OptimizationReport(
            recommendations=recommendations,
            analysis=analysis,
            constraints=constraints,
            metadata={
                'optimization_target': optimization_target,
                'num_candidates': len(candidate_configs),
                'num_valid_recommendations': len(recommendations)
            }
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return report
    
    def suggest_batch_sizes(
        self,
        data_format: str,
        base_config: Dict[str, Any],
        gpu_memory_mb: float,
        min_batch_size: int = 1,
        max_batch_size: int = 512
    ) -> List[int]:
        """Suggest optimal batch sizes for given constraints."""
        if not self.is_trained:
            logger.warning("Models not trained. Using heuristic suggestions.")
            return self._heuristic_batch_sizes(gpu_memory_mb, min_batch_size, max_batch_size)
        
        # Generate candidate batch sizes
        candidates = []
        batch_size = min_batch_size
        while batch_size <= max_batch_size:
            candidates.append(batch_size)
            batch_size = min(batch_size * 2, batch_size + 32)  # Exponential + linear growth
        
        # Evaluate each batch size
        valid_batch_sizes = []
        
        for batch_size in candidates:
            config = {
                'batch_size': batch_size,
                'data_format': data_format,
                **base_config
            }
            
            try:
                features = self._config_to_features(config)
                features_scaled = self.feature_scaler.transform([features])
                
                predicted_memory = self.memory_model.predict(features_scaled)[0]
                
                if predicted_memory <= gpu_memory_mb * 0.8:  # Leave 20% buffer
                    valid_batch_sizes.append(batch_size)
            except Exception as e:
                logger.warning(f"Failed to evaluate batch size {batch_size}: {e}")
        
        return valid_batch_sizes[:10]  # Return top 10
    
    def analyze_trade_offs(self, results: List[ProfilingResult]) -> Dict[str, Any]:
        """Analyze trade-offs between different performance metrics."""
        if not results:
            return {}
        
        df = pd.DataFrame([
            {
                'batch_size': r.batch_size,
                'throughput': r.throughput_samples_per_sec,
                'memory': r.peak_gpu_memory_mb,
                'stability': r.stability_score,
                'data_format': r.data_format
            }
            for r in results
        ])
        
        analysis = {
            'correlations': df[['batch_size', 'throughput', 'memory', 'stability']].corr().to_dict(),
            'pareto_optimal': self._find_pareto_optimal(df),
            'trade_off_curves': self._generate_trade_off_curves(df),
            'format_comparison': self._compare_formats(df),
            'batch_size_analysis': self._analyze_batch_sizes(df)
        }
        
        return analysis
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from profiling results."""
        features = []
        throughput_targets = []
        memory_targets = []
        stability_targets = []
        
        for result in self.training_data:
            try:
                feature_vector = self._result_to_features(result)
                features.append(feature_vector)
                throughput_targets.append(result.throughput_samples_per_sec)
                memory_targets.append(result.peak_gpu_memory_mb)
                stability_targets.append(result.stability_score)
            except Exception as e:
                logger.warning(f"Failed to process training sample: {e}")
        
        return (
            np.array(features),
            np.array(throughput_targets),
            np.array(memory_targets),
            np.array(stability_targets)
        )
    
    def _result_to_features(self, result: ProfilingResult) -> List[float]:
        """Convert a profiling result to feature vector."""
        # Basic features
        features = [
            result.batch_size,
            len(result.data_format),  # Format complexity proxy
            result.loader_config.get('num_workers', 0),
            result.loader_config.get('prefetch_factor', 2),
            int(result.loader_config.get('pin_memory', False)),
            int(result.loader_config.get('drop_last', False)),
        ]
        
        # Format encoding (one-hot)
        format_encodings = {
            'tfrecord': [1, 0, 0],
            'csv': [0, 1, 0],
            'parquet': [0, 0, 1]
        }
        features.extend(format_encodings.get(result.data_format, [0, 0, 0]))
        
        return features
    
    def _config_to_features(self, config: Dict[str, Any]) -> List[float]:
        """Convert configuration to feature vector."""
        features = [
            config.get('batch_size', 32),
            len(config.get('data_format', '')),
            config.get('num_workers', 0),
            config.get('prefetch_factor', 2),
            int(config.get('pin_memory', False)),
            int(config.get('drop_last', False)),
        ]
        
        # Format encoding
        format_encodings = {
            'tfrecord': [1, 0, 0],
            'csv': [0, 1, 0],
            'parquet': [0, 0, 1]
        }
        data_format = config.get('data_format', '')
        features.extend(format_encodings.get(data_format, [0, 0, 0]))
        
        return features
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> OptimizationRecommendation:
        """Evaluate a single configuration."""
        features = self._config_to_features(config)
        features_scaled = self.feature_scaler.transform([features])
        
        # Predict performance
        predicted_throughput = self.throughput_model.predict(features_scaled)[0]
        predicted_memory = self.memory_model.predict(features_scaled)[0]
        predicted_stability = self.stability_model.predict(features_scaled)[0]
        
        # Calculate confidence (based on prediction variance in ensemble)
        throughput_preds = [tree.predict(features_scaled)[0] for tree in self.throughput_model.estimators_[:10]]
        confidence_score = 1.0 - (np.std(throughput_preds) / max(np.mean(throughput_preds), 1e-6))
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Check constraints
        max_memory = constraints.get('max_memory_mb', float('inf'))
        min_throughput = constraints.get('min_throughput', 0)
        
        satisfies_memory = predicted_memory <= max_memory
        satisfies_throughput = predicted_throughput >= min_throughput
        
        # Calculate efficiency scores
        max_possible_throughput = max(r.throughput_samples_per_sec for r in self.training_data)
        max_possible_memory = max(r.peak_gpu_memory_mb for r in self.training_data)
        
        speed_efficiency = predicted_throughput / max_possible_throughput
        memory_efficiency = 1.0 - (predicted_memory / max_possible_memory)
        stability_score = predicted_stability
        
        # Overall score (weighted combination)
        overall_score = (
            0.4 * speed_efficiency +
            0.3 * memory_efficiency +
            0.2 * stability_score +
            0.1 * confidence_score
        )
        
        # Generate reasoning
        reasoning = []
        reasoning.append(f"Predicted throughput: {predicted_throughput:.1f} samples/sec")
        reasoning.append(f"Predicted memory usage: {predicted_memory:.1f} MB")
        reasoning.append(f"Predicted stability: {predicted_stability:.3f}")
        
        if not satisfies_memory:
            reasoning.append(f"⚠️ Exceeds memory constraint ({predicted_memory:.1f} > {max_memory:.1f} MB)")
        if not satisfies_throughput:
            reasoning.append(f"⚠️ Below throughput constraint ({predicted_throughput:.1f} < {min_throughput:.1f})")
        
        return OptimizationRecommendation(
            batch_size=config.get('batch_size', 32),
            data_format=config.get('data_format', 'unknown'),
            loader_config={k: v for k, v in config.items() if k not in ['batch_size', 'data_format']},
            predicted_throughput=predicted_throughput,
            predicted_memory_usage=predicted_memory,
            predicted_stability=predicted_stability,
            confidence_score=confidence_score,
            reasoning=reasoning,
            memory_efficiency_score=memory_efficiency,
            speed_efficiency_score=speed_efficiency,
            stability_score=stability_score,
            overall_score=overall_score,
            satisfies_memory_constraint=satisfies_memory,
            satisfies_throughput_constraint=satisfies_throughput
        )
    
    def _rank_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        optimization_target: str
    ) -> List[OptimizationRecommendation]:
        """Rank recommendations based on optimization target."""
        if optimization_target == 'speed':
            key_func = lambda r: (-r.speed_efficiency_score, -r.overall_score)
        elif optimization_target == 'memory':
            key_func = lambda r: (-r.memory_efficiency_score, -r.overall_score)
        elif optimization_target == 'stability':
            key_func = lambda r: (-r.stability_score, -r.overall_score)
        else:  # balanced
            key_func = lambda r: -r.overall_score
        
        sorted_recommendations = sorted(recommendations, key=key_func)
        
        # Assign ranks
        for i, rec in enumerate(sorted_recommendations):
            rec.rank = i + 1
        
        return sorted_recommendations
    
    def _generate_analysis(
        self,
        recommendations: List[OptimizationRecommendation],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis of recommendations."""
        if not recommendations:
            return {}
        
        valid_recommendations = [
            r for r in recommendations
            if r.satisfies_memory_constraint and r.satisfies_throughput_constraint
        ]
        
        analysis = {
            'total_candidates': len(recommendations),
            'valid_candidates': len(valid_recommendations),
            'constraint_satisfaction_rate': len(valid_recommendations) / len(recommendations),
            'top_recommendation': {
                'batch_size': recommendations[0].batch_size,
                'predicted_throughput': recommendations[0].predicted_throughput,
                'predicted_memory': recommendations[0].predicted_memory_usage,
                'confidence': recommendations[0].confidence_score
            },
            'performance_range': {
                'throughput': {
                    'min': min(r.predicted_throughput for r in recommendations),
                    'max': max(r.predicted_throughput for r in recommendations),
                    'avg': np.mean([r.predicted_throughput for r in recommendations])
                },
                'memory': {
                    'min': min(r.predicted_memory_usage for r in recommendations),
                    'max': max(r.predicted_memory_usage for r in recommendations),
                    'avg': np.mean([r.predicted_memory_usage for r in recommendations])
                }
            }
        }
        
        return analysis
    
    def _heuristic_batch_sizes(
        self,
        gpu_memory_mb: float,
        min_batch_size: int,
        max_batch_size: int
    ) -> List[int]:
        """Generate heuristic batch size suggestions."""
        # Simple heuristic based on GPU memory
        if gpu_memory_mb < 4000:  # < 4GB
            candidates = [1, 2, 4, 8, 16]
        elif gpu_memory_mb < 8000:  # < 8GB
            candidates = [8, 16, 32, 64]
        elif gpu_memory_mb < 16000:  # < 16GB
            candidates = [16, 32, 64, 128, 256]
        else:  # >= 16GB
            candidates = [32, 64, 128, 256, 512]
        
        # Filter by constraints
        candidates = [b for b in candidates if min_batch_size <= b <= max_batch_size]
        
        return candidates[:10]
    
    def _find_pareto_optimal(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find Pareto optimal solutions."""
        # Simple 2D Pareto front: throughput vs memory
        pareto_optimal = []
        
        # Sort by throughput descending
        df_sorted = df.sort_values('throughput', ascending=False)
        
        min_memory = float('inf')
        for _, row in df_sorted.iterrows():
            if row['memory'] < min_memory:
                min_memory = row['memory']
                pareto_optimal.append(row.to_dict())
        
        return pareto_optimal
    
    def _generate_trade_off_curves(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trade-off curve data."""
        return {
            'throughput_vs_memory': df[['throughput', 'memory']].to_dict('records'),
            'batch_size_vs_throughput': df[['batch_size', 'throughput']].to_dict('records'),
            'batch_size_vs_memory': df[['batch_size', 'memory']].to_dict('records')
        }
    
    def _compare_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across data formats."""
        format_stats = {}
        
        for data_format in df['data_format'].unique():
            format_df = df[df['data_format'] == data_format]
            format_stats[data_format] = {
                'avg_throughput': format_df['throughput'].mean(),
                'avg_memory': format_df['memory'].mean(),
                'count': len(format_df)
            }
        
        return format_stats
    
    def _analyze_batch_sizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze batch size effects."""
        batch_analysis = {}
        
        for batch_size in sorted(df['batch_size'].unique()):
            batch_df = df[df['batch_size'] == batch_size]
            batch_analysis[str(batch_size)] = {
                'avg_throughput': batch_df['throughput'].mean(),
                'avg_memory': batch_df['memory'].mean(),
                'count': len(batch_df)
            }
        
        return batch_analysis
    
    def save_model(self, filepath: Path):
        """Save trained models to disk."""
        if not self.is_trained:
            raise RuntimeError("No trained models to save")
        
        import joblib
        
        model_data = {
            'throughput_model': self.throughput_model,
            'memory_model': self.memory_model,
            'stability_model': self.stability_model,
            'feature_scaler': self.feature_scaler,
            'training_data_size': len(self.training_data)
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved models to: {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained models from disk."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.throughput_model = model_data['throughput_model']
        self.memory_model = model_data['memory_model']
        self.stability_model = model_data['stability_model']
        self.feature_scaler = model_data['feature_scaler']
        
        self.is_trained = True
        logger.info(f"Loaded models from: {filepath}")
        logger.info(f"Models trained on {model_data.get('training_data_size', 'unknown')} samples") 