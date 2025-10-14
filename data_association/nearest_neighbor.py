import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional, Type
from data_association.data_association import DataAssociation
from data_association.association_gate import cubical_gate, ellipsoidal_gate
from measurement_models.measurement_model import MeasurementModel


class NearestNeighbor(DataAssociation):
    """
    Nearest Neighbor data association algorithm.
    
    Associates each measurement to the closest track based on distance metric.
    """
    
    def __init__(self, gate_threshold: float = 5.0, use_ellipsoidal_gate: bool = True):
        """
        Initialize Nearest Neighbor data association.
        
        Args:
            gate_threshold: Threshold for gating measurements
            use_ellipsoidal_gate: Whether to use ellipsoidal or cubical gating
        """
        super().__init__()
        self.gate_threshold = gate_threshold
        self.use_ellipsoidal_gate = use_ellipsoidal_gate
    
    def associate(
        self, 
        track_states: List[jnp.ndarray], 
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using nearest neighbor.
        
        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        if not track_states or not measurements:
            return {}
            
        associations = {}
        used_measurements = set()
        
        # For each track, find the closest valid measurement
        for track_idx, track_state in enumerate(track_states):
            best_measurement_idx = None
            best_distance = float('inf')
            
            for meas_idx, (measurement, measurement_model, R) in enumerate(measurements):
                if meas_idx in used_measurements:
                    continue
                    
                # Apply gating
                if self.use_ellipsoidal_gate:
                    if not ellipsoidal_gate(track_state, measurement, measurement_model, self.gate_threshold):
                        continue
                else:
                    if not cubical_gate(track_state, measurement, measurement_model, self.gate_threshold):
                        continue
                
                # Calculate distance (using innovation covariance for proper weighting)
                distance = self._calculate_distance(track_state, measurement, measurement_model, R)
                
                if distance < best_distance:
                    best_distance = distance
                    best_measurement_idx = meas_idx
            
            # Associate if a valid measurement was found
            if best_measurement_idx is not None:
                associations[track_idx] = best_measurement_idx
                used_measurements.add(best_measurement_idx)
                
        return associations
    
    def _calculate_distance(
        self, 
        state: jnp.ndarray, 
        measurement: jnp.ndarray, 
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray
    ) -> float:
        """
        Calculate Mahalanobis distance between predicted and actual measurement.
        
        Args:
            state: Track state vector
            measurement: Actual measurement
            measurement_model: Measurement model class
            R: Measurement noise covariance
            
        Returns:
            Mahalanobis distance
        """
        predicted_measurement = measurement_model.predict_measurement(state)
        innovation = measurement - predicted_measurement
        
        # Use measurement noise covariance for distance calculation
        try:
            inv_R = jnp.linalg.inv(R)
            distance = jnp.sqrt(innovation.T @ inv_R @ innovation)
        except:
            # Fallback to Euclidean distance if covariance inversion fails
            distance = jnp.linalg.norm(innovation)
            
        return float(distance)


class GlobalNearestNeighbor(DataAssociation):
    """
    Global Nearest Neighbor data association algorithm.
    
    Finds the globally optimal assignment of measurements to tracks
    using the Hungarian algorithm (or similar optimization).
    """
    
    def __init__(self, gate_threshold: float = 5.0, use_ellipsoidal_gate: bool = True):
        """
        Initialize Global Nearest Neighbor data association.
        
        Args:
            gate_threshold: Threshold for gating measurements
            use_ellipsoidal_gate: Whether to use ellipsoidal or cubical gating
        """
        super().__init__()
        self.gate_threshold = gate_threshold
        self.use_ellipsoidal_gate = use_ellipsoidal_gate
    
    def associate(
        self, 
        track_states: List[jnp.ndarray], 
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using global nearest neighbor.
        
        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        if not track_states or not measurements:
            return {}
            
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(track_states, measurements)
        
        # Solve assignment problem using simple greedy approach
        # (In practice, you'd use Hungarian algorithm from scipy.optimize.linear_sum_assignment)
        associations = self._solve_assignment(cost_matrix)
        
        return associations
    
    def _build_cost_matrix(
        self, 
        track_states: List[jnp.ndarray], 
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]
    ) -> jnp.ndarray:
        """
        Build cost matrix for assignment problem.
        
        Args:
            track_states: List of track state vectors
            measurements: List of measurement tuples
            
        Returns:
            Cost matrix where cost_matrix[i,j] is cost of assigning track i to measurement j
        """
        n_tracks = len(track_states)
        n_measurements = len(measurements)
        
        # Initialize with large costs (indicating invalid associations)
        cost_matrix = jnp.full((n_tracks, n_measurements), 1e6)
        
        for track_idx, track_state in enumerate(track_states):
            for meas_idx, (measurement, measurement_model, R) in enumerate(measurements):
                # Apply gating
                if self.use_ellipsoidal_gate:
                    gate_valid = ellipsoidal_gate(track_state, measurement, measurement_model, self.gate_threshold)
                else:
                    gate_valid = cubical_gate(track_state, measurement, measurement_model, self.gate_threshold)
                
                if gate_valid:
                    # Calculate cost (distance)
                    cost = self._calculate_distance(track_state, measurement, measurement_model, R)
                    cost_matrix = cost_matrix.at[track_idx, meas_idx].set(cost)
        
        return cost_matrix
    
    def _calculate_distance(
        self, 
        state: jnp.ndarray, 
        measurement: jnp.ndarray, 
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray
    ) -> float:
        """Calculate Mahalanobis distance between predicted and actual measurement."""
        predicted_measurement = measurement_model.predict_measurement(state)
        innovation = measurement - predicted_measurement
        
        try:
            inv_R = jnp.linalg.inv(R)
            distance = jnp.sqrt(innovation.T @ inv_R @ innovation)
        except:
            distance = jnp.linalg.norm(innovation)
            
        return float(distance)
    
    def _solve_assignment(self, cost_matrix: jnp.ndarray) -> Dict[int, int]:
        """
        Solve assignment problem using greedy approach.
        
        Note: This is a simplified implementation. For production use,
        consider using scipy.optimize.linear_sum_assignment for optimal results.
        
        Args:
            cost_matrix: Cost matrix for assignment
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        associations = {}
        used_measurements = set()
        
        n_tracks, n_measurements = cost_matrix.shape
        
        # Greedy assignment: repeatedly find minimum cost valid assignment
        for _ in range(min(n_tracks, n_measurements)):
            min_cost = float('inf')
            best_track = -1
            best_measurement = -1
            
            for track_idx in range(n_tracks):
                if track_idx in associations:
                    continue
                    
                for meas_idx in range(n_measurements):
                    if meas_idx in used_measurements:
                        continue
                        
                    cost = cost_matrix[track_idx, meas_idx]
                    if cost < min_cost and cost < 1e5:  # Valid association threshold
                        min_cost = cost
                        best_track = track_idx
                        best_measurement = meas_idx
            
            if best_track >= 0 and best_measurement >= 0:
                associations[best_track] = best_measurement
                used_measurements.add(best_measurement)
            else:
                break  # No more valid associations
                
        return associations
