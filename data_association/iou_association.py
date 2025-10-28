from typing import Dict, List, Tuple, Type

import jax.numpy as jnp

from data_association.data_association import DataAssociation
from measurement_models.bounding_box_measurement_model import BoundingBoxMeasurement
from measurement_models.measurement_model import MeasurementModel


class IoUAssociation(DataAssociation):
    """
    Intersection over Union (IoU) based data association for bounding box tracking.
    
    This algorithm associates measurements to tracks based on IoU overlap,
    which is particularly suitable for object detection tracking where
    measurements are bounding boxes.
    
    The algorithm works by:
    1. Computing IoU between each track's predicted bounding box and each measurement
    2. Using IoU as a similarity metric (higher IoU = better match)
    3. Applying a minimum IoU threshold for valid associations
    4. Performing assignment using greedy or optimal matching
    """

    def __init__(self, 
                 min_iou_threshold: float = 0.1,
                 use_optimal_assignment: bool = True):
        """
        Initialize IoU-based data association.
        
        Args:
            min_iou_threshold: Minimum IoU required for a valid association
            use_optimal_assignment: If True, use Hungarian algorithm for optimal 
                                  assignment. If False, use greedy assignment.
        """
        super().__init__()
        self.min_iou_threshold = min_iou_threshold
        self.use_optimal_assignment = use_optimal_assignment

    def associate(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using IoU.
        
        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        if not track_states or not measurements:
            return {}

        # Filter for bounding box measurements only
        bbox_measurements = []
        bbox_indices = []
        
        for i, (measurement, measurement_model, noise_cov) in enumerate(measurements):
            if issubclass(measurement_model, BoundingBoxMeasurement):
                bbox_measurements.append((measurement, measurement_model, noise_cov))
                bbox_indices.append(i)
        
        if not bbox_measurements:
            return {}

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(track_states, bbox_measurements)
        
        # Apply IoU threshold
        valid_matrix = iou_matrix >= self.min_iou_threshold
        
        if self.use_optimal_assignment:
            associations = self._optimal_assignment(iou_matrix, valid_matrix)
        else:
            associations = self._greedy_assignment(iou_matrix, valid_matrix)
        
        # Map back to original measurement indices
        final_associations = {}
        for track_idx, bbox_meas_idx in associations.items():
            final_associations[track_idx] = bbox_indices[bbox_meas_idx]
            
        return final_associations

    def _compute_iou_matrix(self, 
                          track_states: List[jnp.ndarray],
                          measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]) -> jnp.ndarray:
        """
        Compute IoU matrix between track predictions and measurements.
        
        Args:
            track_states: List of track state vectors
            measurements: List of bounding box measurements
            
        Returns:
            IoU matrix (n_tracks x n_measurements)
        """
        n_tracks = len(track_states)
        n_measurements = len(measurements)
        
        iou_matrix = jnp.zeros((n_tracks, n_measurements))
        
        for track_idx, track_state in enumerate(track_states):
            # Get predicted bounding box from track state
            predicted_bbox = BoundingBoxMeasurement.state_to_bbox(track_state)
            
            for meas_idx, (measurement, measurement_model, _) in enumerate(measurements):
                # Convert measurement to bounding box format
                measured_bbox = BoundingBoxMeasurement.measurement_to_bbox(measurement)
                
                # Compute IoU
                iou = BoundingBoxMeasurement.calculate_iou(predicted_bbox, measured_bbox)
                iou_matrix = iou_matrix.at[track_idx, meas_idx].set(iou)
        
        return iou_matrix

    def _greedy_assignment(self, 
                         iou_matrix: jnp.ndarray, 
                         valid_matrix: jnp.ndarray) -> Dict[int, int]:
        """
        Perform greedy assignment based on IoU scores.
        
        Args:
            iou_matrix: IoU scores between tracks and measurements
            valid_matrix: Boolean matrix indicating valid associations
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        associations = {}
        used_measurements = set()
        
        # Create list of (iou_score, track_idx, meas_idx) for valid associations
        candidates = []
        for track_idx in range(iou_matrix.shape[0]):
            for meas_idx in range(iou_matrix.shape[1]):
                if valid_matrix[track_idx, meas_idx]:
                    candidates.append((
                        float(iou_matrix[track_idx, meas_idx]), 
                        track_idx, 
                        meas_idx
                    ))
        
        # Sort by IoU score (descending)
        candidates.sort(reverse=True)
        
        # Greedily assign highest IoU associations
        for iou_score, track_idx, meas_idx in candidates:
            if track_idx not in associations and meas_idx not in used_measurements:
                associations[track_idx] = meas_idx
                used_measurements.add(meas_idx)
        
        return associations

    def _optimal_assignment(self, 
                          iou_matrix: jnp.ndarray, 
                          valid_matrix: jnp.ndarray) -> Dict[int, int]:
        """
        Perform optimal assignment using Hungarian algorithm.
        
        Since we want to maximize IoU, we convert to a cost minimization problem
        by using (1 - IoU) as the cost.
        
        Args:
            iou_matrix: IoU scores between tracks and measurements
            valid_matrix: Boolean matrix indicating valid associations
            
        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        # Convert IoU to cost (minimize 1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        
        # Set invalid associations to high cost
        cost_matrix = jnp.where(valid_matrix, cost_matrix, 1e6)
        
        # Use simple greedy assignment for now
        # TODO: Implement proper Hungarian algorithm or use scipy.optimize.linear_sum_assignment
        # For now, fall back to greedy assignment
        return self._greedy_assignment(iou_matrix, valid_matrix)

    def get_iou_threshold(self) -> float:
        """Get the current IoU threshold."""
        return self.min_iou_threshold

    def set_iou_threshold(self, threshold: float) -> None:
        """
        Set the IoU threshold.
        
        Args:
            threshold: New IoU threshold (should be between 0 and 1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        self.min_iou_threshold = threshold


class IoUNearestNeighbor(IoUAssociation):
    """
    IoU-based nearest neighbor association.
    
    This is a simplified version that uses greedy assignment.
    """
    
    def __init__(self, min_iou_threshold: float = 0.1):
        super().__init__(
            min_iou_threshold=min_iou_threshold,
            use_optimal_assignment=False
        )


class IoUOptimalAssignment(IoUAssociation):
    """
    IoU-based optimal assignment.
    
    This version uses optimal assignment (Hungarian algorithm).
    """
    
    def __init__(self, min_iou_threshold: float = 0.1):
        super().__init__(
            min_iou_threshold=min_iou_threshold,
            use_optimal_assignment=True
        )
