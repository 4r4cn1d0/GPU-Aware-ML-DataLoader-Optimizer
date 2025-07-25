"""
Learning engine that implements advanced algorithms for recognizing optimal data loading patterns.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

from .profiler import ProfilingResult, ProfilingSession


logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a learned data loading pattern."""
    
    pattern_id: str
    pattern_type: str  # 'optimal', 'suboptimal', 'anomaly'
    
    # Pattern characteristics
    batch_size_range: Tuple[int, int]
    data_formats: List[str]
    performance_profile: Dict[str, float]
    
    # Statistical properties
    confidence: float
    frequency: int
    last_seen: datetime
    
    # Context
    conditions: Dict[str, Any]  # GPU memory, data size, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningState:
    """Represents the current state of the learning engine."""
    
    patterns: List[Pattern] = field(default_factory=list)
    anomalies: List[ProfilingResult] = field(default_factory=list)
    performance_baselines: Dict[str, float] = field(default_factory=dict)
    
    # Learning statistics
    total_samples_processed: int = 0
    last_learning_update: Optional[datetime] = None
    learning_history: List[Dict[str, Any]] = field(default_factory=list)


class LearningEngine:
    """
    Advanced learning engine for recognizing optimal data loading patterns.
    
    Features:
    - Pattern recognition using clustering and statistical analysis
    - Anomaly detection for identifying performance outliers
    - Adaptive learning with concept drift detection
    - Temporal pattern analysis
    - Performance forecasting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'min_pattern_samples': 10,
            'anomaly_threshold': 0.1,
            'pattern_similarity_threshold': 0.8,
            'max_patterns': 100,
            'learning_rate': 0.1,
            'temporal_window_hours': 24
        }
        
        self.state = LearningState()
        
        # ML models
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Temporal analysis
        self.temporal_window = deque(maxlen=1000)
        
        logger.info("LearningEngine initialized")
    
    def learn_from_session(self, session: ProfilingSession):
        """Learn patterns from a complete profiling session."""
        if not session.results:
            logger.warning("No results in session to learn from")
            return
        
        logger.info(f"Learning from session: {session.session_id} ({len(session.results)} results)")
        
        # Add results to temporal window
        for result in session.results:
            self.temporal_window.append(result)
        
        # Update learning state
        self.state.total_samples_processed += len(session.results)
        
        # Perform learning steps
        self._extract_patterns(session.results)
        self._detect_anomalies(session.results)
        self._update_baselines(session.results)
        self._analyze_temporal_trends()
        
        # Record learning update
        self.state.last_learning_update = datetime.now()
        self.state.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'session_id': session.session_id,
            'samples_added': len(session.results),
            'total_patterns': len(self.state.patterns),
            'total_anomalies': len(self.state.anomalies)
        })
        
        # Cleanup old patterns
        self._cleanup_patterns()
        
        logger.info(f"Learning complete. Patterns: {len(self.state.patterns)}, "
                   f"Anomalies: {len(self.state.anomalies)}")
    
    def predict_performance(
        self,
        batch_size: int,
        data_format: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Predict performance based on learned patterns."""
        context = context or {}
        
        # Find matching patterns
        matching_patterns = self._find_matching_patterns(batch_size, data_format, context)
        
        if not matching_patterns:
            # Use baseline predictions
            return self._baseline_prediction(batch_size, data_format)
        
        # Weight predictions by pattern confidence and frequency
        weighted_prediction = defaultdict(float)
        total_weight = 0
        
        for pattern in matching_patterns:
            weight = pattern.confidence * np.log(1 + pattern.frequency)
            
            for metric, value in pattern.performance_profile.items():
                weighted_prediction[metric] += weight * value
            
            total_weight += weight
        
        # Normalize predictions
        if total_weight > 0:
            for metric in weighted_prediction:
                weighted_prediction[metric] /= total_weight
        
        return dict(weighted_prediction)
    
    def recommend_configuration(
        self,
        target_metric: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Recommend optimal configuration based on learned patterns."""
        constraints = constraints or {}
        
        # Find patterns that satisfy constraints
        viable_patterns = []
        
        for pattern in self.state.patterns:
            if pattern.pattern_type != 'optimal':
                continue
            
            # Check constraints
            satisfies_constraints = True
            
            if 'max_memory_mb' in constraints:
                pattern_memory = pattern.performance_profile.get('peak_gpu_memory_mb', 0)
                if pattern_memory > constraints['max_memory_mb']:
                    satisfies_constraints = False
            
            if 'min_throughput' in constraints:
                pattern_throughput = pattern.performance_profile.get('throughput_samples_per_sec', 0)
                if pattern_throughput < constraints['min_throughput']:
                    satisfies_constraints = False
            
            if satisfies_constraints:
                viable_patterns.append(pattern)
        
        if not viable_patterns:
            return None
        
        # Rank patterns by target metric and confidence
        def pattern_score(pattern):
            metric_value = pattern.performance_profile.get(target_metric, 0)
            confidence_bonus = pattern.confidence * 0.1
            frequency_bonus = np.log(1 + pattern.frequency) * 0.05
            return metric_value + confidence_bonus + frequency_bonus
        
        best_pattern = max(viable_patterns, key=pattern_score)
        
        # Generate configuration recommendation
        recommendation = {
            'batch_size': (best_pattern.batch_size_range[0] + best_pattern.batch_size_range[1]) // 2,
            'data_format': best_pattern.data_formats[0] if best_pattern.data_formats else 'tfrecord',
            'confidence': best_pattern.confidence,
            'expected_performance': best_pattern.performance_profile,
            'pattern_id': best_pattern.pattern_id
        }
        
        # Add learned configuration parameters
        if 'conditions' in best_pattern.metadata:
            recommendation.update(best_pattern.metadata['conditions'])
        
        return recommendation
    
    def detect_concept_drift(self, recent_results: List[ProfilingResult]) -> Dict[str, Any]:
        """Detect if there has been concept drift in performance patterns."""
        if len(recent_results) < 10:
            return {'drift_detected': False, 'confidence': 0.0}
        
        if not self.state.performance_baselines:
            return {'drift_detected': False, 'confidence': 0.0}
        
        # Compare recent performance to historical baselines
        drift_signals = {}
        
        # Calculate recent performance metrics
        recent_metrics = self._calculate_aggregate_metrics(recent_results)
        
        for metric, recent_value in recent_metrics.items():
            if metric in self.state.performance_baselines:
                baseline_value = self.state.performance_baselines[metric]
                
                # Calculate relative change
                if baseline_value > 0:
                    relative_change = abs(recent_value - baseline_value) / baseline_value
                    drift_signals[metric] = relative_change
        
        # Overall drift score
        drift_scores = list(drift_signals.values())
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        drift_threshold = 0.2  # 20% change threshold
        drift_detected = overall_drift > drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'confidence': min(overall_drift / drift_threshold, 1.0),
            'metric_drifts': drift_signals,
            'overall_drift_score': overall_drift
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate insights from learned patterns."""
        if not self.state.patterns:
            return {'insights': [], 'summary': 'No patterns learned yet'}
        
        insights = []
        
        # Analyze optimal patterns
        optimal_patterns = [p for p in self.state.patterns if p.pattern_type == 'optimal']
        
        if optimal_patterns:
            # Most common optimal batch size range
            batch_ranges = [p.batch_size_range for p in optimal_patterns]
            avg_min_batch = np.mean([r[0] for r in batch_ranges])
            avg_max_batch = np.mean([r[1] for r in batch_ranges])
            
            insights.append(f"Optimal batch sizes typically range from {avg_min_batch:.0f} to {avg_max_batch:.0f}")
            
            # Best performing format
            format_performance = defaultdict(list)
            for pattern in optimal_patterns:
                for fmt in pattern.data_formats:
                    throughput = pattern.performance_profile.get('throughput_samples_per_sec', 0)
                    format_performance[fmt].append(throughput)
            
            if format_performance:
                avg_performance = {fmt: np.mean(perfs) for fmt, perfs in format_performance.items()}
                best_format = max(avg_performance.keys(), key=lambda k: avg_performance[k])
                insights.append(f"Best performing data format: {best_format}")
        
        # Memory usage patterns
        memory_usages = []
        for pattern in self.state.patterns:
            memory = pattern.performance_profile.get('peak_gpu_memory_mb', 0)
            if memory > 0:
                memory_usages.append(memory)
        
        if memory_usages:
            avg_memory = np.mean(memory_usages)
            insights.append(f"Average GPU memory usage: {avg_memory:.0f} MB")
        
        # Anomaly analysis
        if self.state.anomalies:
            insights.append(f"Detected {len(self.state.anomalies)} performance anomalies")
        
        summary = f"Learned {len(self.state.patterns)} patterns from {self.state.total_samples_processed} samples"
        
        return {
            'insights': insights,
            'summary': summary,
            'pattern_count': len(self.state.patterns),
            'anomaly_count': len(self.state.anomalies),
            'last_update': self.state.last_learning_update.isoformat() if self.state.last_learning_update else None
        }
    
    def _extract_patterns(self, results: List[ProfilingResult]):
        """Extract patterns from profiling results using clustering."""
        if len(results) < self.config['min_pattern_samples']:
            return
        
        # Convert results to feature vectors
        features = []
        for result in results:
            feature_vector = [
                result.batch_size,
                result.throughput_samples_per_sec,
                result.peak_gpu_memory_mb,
                result.stability_score,
                len(result.data_format),  # Format complexity proxy
                result.avg_load_time,
                result.cpu_utilization,
                result.gpu_utilization
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Perform clustering
        n_clusters = min(5, len(results) // 2)  # Adaptive cluster count
        self.clusterer.n_clusters = n_clusters
        cluster_labels = self.clusterer.fit_predict(features_pca)
        
        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_results = [results[i] for i in range(len(results)) if cluster_labels[i] == cluster_id]
            
            if len(cluster_results) >= 3:  # Minimum cluster size
                pattern = self._create_pattern_from_cluster(cluster_results, cluster_id)
                if pattern:
                    self._add_or_update_pattern(pattern)
    
    def _create_pattern_from_cluster(self, cluster_results: List[ProfilingResult], cluster_id: int) -> Optional[Pattern]:
        """Create a pattern from a cluster of results."""
        if not cluster_results:
            return None
        
        # Calculate aggregate metrics
        batch_sizes = [r.batch_size for r in cluster_results]
        throughputs = [r.throughput_samples_per_sec for r in cluster_results]
        memories = [r.peak_gpu_memory_mb for r in cluster_results]
        stabilities = [r.stability_score for r in cluster_results]
        
        # Determine pattern type based on performance
        avg_throughput = np.mean(throughputs)
        avg_stability = np.mean(stabilities)
        
        # Use percentiles to classify patterns
        throughput_threshold = np.percentile([r.throughput_samples_per_sec for r in self.temporal_window], 75)
        stability_threshold = 0.8
        
        if avg_throughput >= throughput_threshold and avg_stability >= stability_threshold:
            pattern_type = 'optimal'
        elif avg_throughput < np.percentile([r.throughput_samples_per_sec for r in self.temporal_window], 25):
            pattern_type = 'suboptimal'
        else:
            pattern_type = 'typical'
        
        # Get data formats in this cluster
        data_formats = list(set(r.data_format for r in cluster_results))
        
        # Create pattern
        pattern = Pattern(
            pattern_id=f"cluster_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_type=pattern_type,
            batch_size_range=(min(batch_sizes), max(batch_sizes)),
            data_formats=data_formats,
            performance_profile={
                'throughput_samples_per_sec': avg_throughput,
                'peak_gpu_memory_mb': np.mean(memories),
                'stability_score': avg_stability,
                'avg_load_time': np.mean([r.avg_load_time for r in cluster_results]),
                'cpu_utilization': np.mean([r.cpu_utilization for r in cluster_results]),
                'gpu_utilization': np.mean([r.gpu_utilization for r in cluster_results])
            },
            confidence=self._calculate_pattern_confidence(cluster_results),
            frequency=len(cluster_results),
            last_seen=datetime.now(),
            conditions=self._extract_conditions(cluster_results)
        )
        
        return pattern
    
    def _calculate_pattern_confidence(self, results: List[ProfilingResult]) -> float:
        """Calculate confidence score for a pattern."""
        if len(results) < 2:
            return 0.5
        
        # Base confidence on consistency of results
        throughputs = [r.throughput_samples_per_sec for r in results]
        cv = np.std(throughputs) / np.mean(throughputs) if np.mean(throughputs) > 0 else float('inf')
        
        # Higher consistency = higher confidence
        consistency_score = 1.0 / (1.0 + cv)
        
        # Boost confidence with more samples
        sample_boost = min(1.0, len(results) / 20.0)
        
        # Combine scores
        confidence = 0.7 * consistency_score + 0.3 * sample_boost
        
        return max(0.1, min(1.0, confidence))
    
    def _extract_conditions(self, results: List[ProfilingResult]) -> Dict[str, Any]:
        """Extract common conditions from results."""
        conditions = {}
        
        # Average resource usage
        conditions['avg_gpu_memory_mb'] = np.mean([r.avg_gpu_memory_mb for r in results])
        conditions['avg_cpu_utilization'] = np.mean([r.cpu_utilization for r in results])
        
        # Common configuration elements
        loader_configs = [r.loader_config for r in results]
        if loader_configs:
            # Find common configuration keys
            common_keys = set(loader_configs[0].keys())
            for config in loader_configs[1:]:
                common_keys &= set(config.keys())
            
            # Extract common values
            for key in common_keys:
                values = [config[key] for config in loader_configs]
                if len(set(values)) == 1:  # All values are the same
                    conditions[key] = values[0]
        
        return conditions
    
    def _add_or_update_pattern(self, new_pattern: Pattern):
        """Add a new pattern or update existing similar pattern."""
        # Check for similar existing patterns
        similar_pattern = None
        
        for existing_pattern in self.state.patterns:
            if self._patterns_similar(existing_pattern, new_pattern):
                similar_pattern = existing_pattern
                break
        
        if similar_pattern:
            # Update existing pattern
            similar_pattern.frequency += new_pattern.frequency
            similar_pattern.last_seen = new_pattern.last_seen
            
            # Update performance profile (weighted average)
            total_freq = similar_pattern.frequency + new_pattern.frequency
            for metric in similar_pattern.performance_profile:
                if metric in new_pattern.performance_profile:
                    old_value = similar_pattern.performance_profile[metric]
                    new_value = new_pattern.performance_profile[metric]
                    
                    weighted_value = (
                        old_value * similar_pattern.frequency +
                        new_value * new_pattern.frequency
                    ) / total_freq
                    
                    similar_pattern.performance_profile[metric] = weighted_value
        else:
            # Add new pattern
            self.state.patterns.append(new_pattern)
    
    def _patterns_similar(self, pattern1: Pattern, pattern2: Pattern) -> bool:
        """Check if two patterns are similar."""
        # Check batch size overlap
        range1 = pattern1.batch_size_range
        range2 = pattern2.batch_size_range
        
        overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
        union = max(range1[1], range2[1]) - min(range1[0], range2[0])
        
        batch_similarity = overlap / union if union > 0 else 0
        
        # Check data format overlap
        formats1 = set(pattern1.data_formats)
        formats2 = set(pattern2.data_formats)
        
        format_similarity = len(formats1 & formats2) / len(formats1 | formats2) if formats1 | formats2 else 1
        
        # Check performance similarity
        perf_similarity = self._calculate_performance_similarity(
            pattern1.performance_profile,
            pattern2.performance_profile
        )
        
        # Overall similarity
        overall_similarity = (batch_similarity + format_similarity + perf_similarity) / 3
        
        return overall_similarity >= self.config['pattern_similarity_threshold']
    
    def _calculate_performance_similarity(self, perf1: Dict[str, float], perf2: Dict[str, float]) -> float:
        """Calculate similarity between performance profiles."""
        common_metrics = set(perf1.keys()) & set(perf2.keys())
        
        if not common_metrics:
            return 0.0
        
        similarities = []
        
        for metric in common_metrics:
            val1, val2 = perf1[metric], perf2[metric]
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                # Relative similarity
                similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _detect_anomalies(self, results: List[ProfilingResult]):
        """Detect anomalous performance results."""
        if len(results) < 10:  # Need minimum samples for anomaly detection
            return
        
        # Extract features for anomaly detection
        features = []
        for result in results:
            feature_vector = [
                result.throughput_samples_per_sec,
                result.peak_gpu_memory_mb,
                result.avg_load_time,
                result.stability_score,
                result.cpu_utilization,
                result.gpu_utilization
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Fit anomaly detector
        anomaly_labels = self.anomaly_detector.fit_predict(features)
        
        # Collect anomalies
        for i, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly
                self.state.anomalies.append(results[i])
        
        # Keep only recent anomalies
        cutoff_time = datetime.now() - timedelta(hours=self.config['temporal_window_hours'])
        self.state.anomalies = [
            a for a in self.state.anomalies
            if datetime.fromtimestamp(a.timestamp) > cutoff_time
        ]
    
    def _update_baselines(self, results: List[ProfilingResult]):
        """Update performance baselines."""
        if not results:
            return
        
        # Calculate current performance metrics
        current_metrics = self._calculate_aggregate_metrics(results)
        
        # Update baselines with exponential moving average
        alpha = self.config['learning_rate']
        
        for metric, value in current_metrics.items():
            if metric in self.state.performance_baselines:
                self.state.performance_baselines[metric] = (
                    alpha * value + (1 - alpha) * self.state.performance_baselines[metric]
                )
            else:
                self.state.performance_baselines[metric] = value
    
    def _calculate_aggregate_metrics(self, results: List[ProfilingResult]) -> Dict[str, float]:
        """Calculate aggregate performance metrics."""
        if not results:
            return {}
        
        return {
            'avg_throughput': np.mean([r.throughput_samples_per_sec for r in results]),
            'avg_memory': np.mean([r.peak_gpu_memory_mb for r in results]),
            'avg_stability': np.mean([r.stability_score for r in results]),
            'avg_load_time': np.mean([r.avg_load_time for r in results]),
            'avg_cpu_util': np.mean([r.cpu_utilization for r in results]),
            'avg_gpu_util': np.mean([r.gpu_utilization for r in results])
        }
    
    def _analyze_temporal_trends(self):
        """Analyze temporal trends in performance."""
        if len(self.temporal_window) < 20:
            return
        
        # Extract time series data
        recent_results = list(self.temporal_window)[-50:]  # Last 50 results
        timestamps = [r.timestamp for r in recent_results]
        throughputs = [r.throughput_samples_per_sec for r in recent_results]
        
        # Simple trend analysis
        if len(throughputs) >= 10:
            # Linear regression to detect trends
            x = np.arange(len(throughputs))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, throughputs)
            
            # Store trend information
            trend_info = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'improving' if slope > 0 else 'degrading' if slope < 0 else 'stable'
            }
            
            # Add to learning history
            if 'temporal_trends' not in self.state.metadata:
                self.state.metadata = {}
            self.state.metadata['temporal_trends'] = trend_info
    
    def _find_matching_patterns(
        self,
        batch_size: int,
        data_format: str,
        context: Dict[str, Any]
    ) -> List[Pattern]:
        """Find patterns that match the given configuration."""
        matching_patterns = []
        
        for pattern in self.state.patterns:
            # Check batch size range
            if not (pattern.batch_size_range[0] <= batch_size <= pattern.batch_size_range[1]):
                continue
            
            # Check data format
            if data_format not in pattern.data_formats:
                continue
            
            # Check context conditions (if available)
            context_match = True
            if context and pattern.conditions:
                for key, value in context.items():
                    if key in pattern.conditions:
                        pattern_value = pattern.conditions[key]
                        # Allow 20% tolerance for numeric values
                        if isinstance(value, (int, float)) and isinstance(pattern_value, (int, float)):
                            if abs(value - pattern_value) / max(pattern_value, 1e-6) > 0.2:
                                context_match = False
                                break
                        elif value != pattern_value:
                            context_match = False
                            break
            
            if context_match:
                matching_patterns.append(pattern)
        
        # Sort by confidence and frequency
        matching_patterns.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
        
        return matching_patterns[:5]  # Return top 5 matches
    
    def _baseline_prediction(self, batch_size: int, data_format: str) -> Dict[str, float]:
        """Generate baseline prediction when no patterns match."""
        if not self.state.performance_baselines:
            return {
                'throughput_samples_per_sec': 100.0,
                'peak_gpu_memory_mb': 1000.0,
                'stability_score': 0.8
            }
        
        # Scale baseline by batch size (rough heuristic)
        batch_scale = batch_size / 32.0  # Normalize to batch size 32
        
        return {
            'throughput_samples_per_sec': self.state.performance_baselines.get('avg_throughput', 100) * batch_scale,
            'peak_gpu_memory_mb': self.state.performance_baselines.get('avg_memory', 1000) * batch_scale,
            'stability_score': self.state.performance_baselines.get('avg_stability', 0.8)
        }
    
    def _cleanup_patterns(self):
        """Remove old or low-confidence patterns."""
        if len(self.state.patterns) <= self.config['max_patterns']:
            return
        
        # Sort patterns by score (confidence * frequency * recency)
        current_time = datetime.now()
        
        def pattern_score(pattern):
            recency_hours = (current_time - pattern.last_seen).total_seconds() / 3600
            recency_factor = 1.0 / (1.0 + recency_hours / 24)  # Decay over days
            
            return pattern.confidence * np.log(1 + pattern.frequency) * recency_factor
        
        # Keep only top patterns
        self.state.patterns.sort(key=pattern_score, reverse=True)
        self.state.patterns = self.state.patterns[:self.config['max_patterns']]
    
    def save_state(self, filepath: Path):
        """Save learning state to disk."""
        state_dict = {
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'batch_size_range': p.batch_size_range,
                    'data_formats': p.data_formats,
                    'performance_profile': p.performance_profile,
                    'confidence': p.confidence,
                    'frequency': p.frequency,
                    'last_seen': p.last_seen.isoformat(),
                    'conditions': p.conditions,
                    'metadata': p.metadata
                }
                for p in self.state.patterns
            ],
            'performance_baselines': self.state.performance_baselines,
            'total_samples_processed': self.state.total_samples_processed,
            'last_learning_update': self.state.last_learning_update.isoformat() if self.state.last_learning_update else None,
            'learning_history': self.state.learning_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Saved learning state to: {filepath}")
    
    def load_state(self, filepath: Path):
        """Load learning state from disk."""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Reconstruct patterns
        patterns = []
        for p_dict in state_dict.get('patterns', []):
            pattern = Pattern(
                pattern_id=p_dict['pattern_id'],
                pattern_type=p_dict['pattern_type'],
                batch_size_range=tuple(p_dict['batch_size_range']),
                data_formats=p_dict['data_formats'],
                performance_profile=p_dict['performance_profile'],
                confidence=p_dict['confidence'],
                frequency=p_dict['frequency'],
                last_seen=datetime.fromisoformat(p_dict['last_seen']),
                conditions=p_dict['conditions'],
                metadata=p_dict.get('metadata', {})
            )
            patterns.append(pattern)
        
        self.state.patterns = patterns
        self.state.performance_baselines = state_dict.get('performance_baselines', {})
        self.state.total_samples_processed = state_dict.get('total_samples_processed', 0)
        
        if state_dict.get('last_learning_update'):
            self.state.last_learning_update = datetime.fromisoformat(state_dict['last_learning_update'])
        
        self.state.learning_history = state_dict.get('learning_history', [])
        
        logger.info(f"Loaded learning state from: {filepath}")
        logger.info(f"Loaded {len(self.state.patterns)} patterns") 