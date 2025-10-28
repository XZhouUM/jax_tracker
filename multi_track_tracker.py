from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import jax.numpy as jnp

from data_association.iou_association import (IoUNearestNeighbor,
                                              IoUOptimalAssignment)
from data_association.joint_probabilistic_data_association import \
    JointProbabilisticDataAssociation
from data_association.multi_hypothesis_tracking import MultiHypothesisTracking
from data_association.nearest_neighbor import (GlobalNearestNeighbor,
                                               NearestNeighbor)
from measurement_models.measurement_model import MeasurementModel
from motion_models.motion_model import MotionModel
from track_management import TrackManager


@dataclass
class TrackerConfig:
    """Configuration for the multi-track tracker."""

    motion_model_class: Type[MotionModel]
    data_association_algorithm: str = (
        # "nearest_neighbor", "global_nearest_neighbor", "mht",
        # "jpda", "iou_nearest_neighbor", or "iou_optimal"
        "nearest_neighbor"
    )
    gate_threshold: float = 5.0
    use_ellipsoidal_gate: bool = True
    confirmation_threshold: int = 3
    deletion_threshold: int = 5
    max_tracks: int = 100
    process_noise_scale: float = 1.0
    initial_covariance_scale: float = 10.0


class MultiTrackTracker:
    """
    Multi-track tracker using Kalman filters with configurable data association.

    This tracker manages multiple targets simultaneously, handling track initialization,
    data association, track updates, and track deletion.
    """

    def __init__(self, config: TrackerConfig):
        """
        Initialize the multi-track tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config

        # Initialize track manager
        self.track_manager = TrackManager(
            motion_model_class=config.motion_model_class,
            confirmation_threshold=config.confirmation_threshold,
            deletion_threshold=config.deletion_threshold,
            max_tracks=config.max_tracks,
        )

        # Initialize data association algorithm
        if config.data_association_algorithm == "nearest_neighbor":
            self.data_associator = NearestNeighbor(
                gate_threshold=config.gate_threshold,
                use_ellipsoidal_gate=config.use_ellipsoidal_gate,
            )
        elif config.data_association_algorithm == "global_nearest_neighbor":
            self.data_associator = GlobalNearestNeighbor(
                gate_threshold=config.gate_threshold,
                use_ellipsoidal_gate=config.use_ellipsoidal_gate,
            )
        elif config.data_association_algorithm == "mht":
            self.data_associator = MultiHypothesisTracking(
                gate_threshold=config.gate_threshold,
                use_ellipsoidal_gate=config.use_ellipsoidal_gate,
            )
        elif config.data_association_algorithm == "jpda":
            self.data_associator = JointProbabilisticDataAssociation(
                gate_threshold=config.gate_threshold,
                use_ellipsoidal_gate=config.use_ellipsoidal_gate,
            )
        elif config.data_association_algorithm == "iou_nearest_neighbor":
            self.data_associator = IoUNearestNeighbor(
                min_iou_threshold=config.gate_threshold,
            )
        elif config.data_association_algorithm == "iou_optimal":
            self.data_associator = IoUOptimalAssignment(
                min_iou_threshold=config.gate_threshold,
            )
        else:
            raise ValueError(
                f"Unknown data association algorithm: {config.data_association_algorithm}"
            )

        self.current_time = 0.0
        self.last_update_time = 0.0

    def update(
        self,
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
        dt: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update the tracker with new measurements.

        Args:
            measurements: List of (measurement, measurement_model, noise_covariance) tuples
            dt: Time step (if None, calculated from current_time)
            current_time: Current time (if None, uses internal time)

        Returns:
            Dictionary containing tracking results and statistics
        """
        # Update timing
        if current_time is not None:
            self.current_time = current_time
        else:
            self.current_time += 1.0  # Default time increment

        if dt is None:
            dt = (
                self.current_time - self.last_update_time
                if self.last_update_time > 0
                else 1.0
            )

        # Step 1: Predict all tracks forward in time
        self._predict_tracks(dt)

        # Step 2: Perform data association
        associations = self._associate_measurements(measurements)

        # Step 3: Update associated tracks
        self._update_tracks(measurements, associations)

        # Step 4: Initialize new tracks from unassociated measurements
        new_track_ids = self._initialize_new_tracks(measurements, associations)

        # Step 5: Update track statuses and cleanup
        self._update_track_statuses(associations)
        deleted_track_ids = self.track_manager.cleanup_deleted_tracks()

        # Update last update time
        self.last_update_time = self.current_time

        # Return results
        return {
            "timestamp": self.current_time,
            "dt": dt,
            "num_measurements": len(measurements),
            "associations": associations,
            "new_tracks": new_track_ids,
            "deleted_tracks": deleted_track_ids,
            "track_summary": self.track_manager.get_track_summary(),
            "confirmed_tracks": self.get_confirmed_track_states(),
        }

    def _predict_tracks(self, dt: float) -> None:
        """Predict all active tracks forward in time."""
        Q = self._get_process_noise_matrix(dt)

        for track in self.track_manager.get_all_active_tracks().values():
            track.update(dt, Q, measurements=None)  # Time update only

    def _associate_measurements(
        self,
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks.

        Returns:
            Dictionary mapping track_id -> measurement_index
        """
        if not measurements:
            return {}

        # Get track states for association
        track_states_with_ids = self.track_manager.get_track_states(
            confirmed_only=False
        )

        if not track_states_with_ids:
            return {}

        # Extract just the states for the data associator
        track_states = [state for _, state in track_states_with_ids]

        # Perform association (returns track_index -> measurement_index)
        track_idx_associations = self.data_associator.associate(
            track_states, measurements
        )

        # Convert track indices back to track IDs
        track_id_associations = {}
        for track_idx, meas_idx in track_idx_associations.items():
            track_id = track_states_with_ids[track_idx][0]
            track_id_associations[track_id] = meas_idx

        return track_id_associations

    def _update_tracks(
        self,
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
        associations: Dict[int, int],
    ) -> None:
        """Update tracks with associated measurements."""
        for track_id, meas_idx in associations.items():
            if track_id in self.track_manager.tracks:
                measurement, measurement_model, R = measurements[meas_idx]

                # Create measurement list for the track update
                measurement_list = [(measurement, measurement_model, R)]

                # Perform measurement update (time update was already done in _predict_tracks)
                track = self.track_manager.tracks[track_id]
                track._measurement_update(measurement_list)

    def _initialize_new_tracks(
        self,
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
        associations: Dict[int, int],
    ) -> List[int]:
        """Initialize new tracks from unassociated measurements."""
        associated_measurements = set(associations.values())
        new_track_ids = []

        for meas_idx, (measurement, measurement_model, R) in enumerate(measurements):
            if meas_idx not in associated_measurements:
                # Initialize new track
                track_id = self.track_manager.initialize_track(
                    measurement=measurement,
                    measurement_model=measurement_model,
                    R=R,
                    current_time=self.current_time,
                    initial_covariance_scale=self.config.initial_covariance_scale,
                )
                new_track_ids.append(track_id)

        return new_track_ids

    def _update_track_statuses(self, associations: Dict[int, int]) -> None:
        """Update track statuses based on association results."""
        # Update all active tracks
        for track_id in self.track_manager.get_all_active_tracks().keys():
            associated = track_id in associations
            self.track_manager.update_track_status(
                track_id=track_id, associated=associated, current_time=self.current_time
            )

    def _get_process_noise_matrix(self, dt: float) -> jnp.ndarray:
        """
        Generate process noise covariance matrix.

        The matrix size depends on the motion model:
        - Constant velocity 2D: 4x4 matrix
        - Bounding box constant velocity: 8x8 matrix
        """
        # Use the motion model's process noise matrix method if available
        if hasattr(self.config.motion_model_class, "process_noise_matrix"):
            try:
                # Try with noise_scale parameter
                return self.config.motion_model_class.process_noise_matrix(
                    dt, self.config.process_noise_scale
                )
            except TypeError:
                # Fallback to method without noise_scale
                Q = self.config.motion_model_class.process_noise_matrix(dt)
                return Q * self.config.process_noise_scale

        # Fallback to 2D constant velocity model
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        Q = (
            jnp.array(
                [
                    [dt4 / 4, 0, dt3 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2],
                    [dt3 / 2, 0, dt2, 0],
                    [0, dt3 / 2, 0, dt2],
                ]
            )
            * self.config.process_noise_scale
        )

        return Q

    def get_confirmed_track_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Get states of all confirmed tracks.

        Returns:
            Dictionary mapping track_id to track state information
        """
        confirmed_tracks = self.track_manager.get_confirmed_tracks()
        track_states = {}

        for track_id, track in confirmed_tracks.items():
            info = self.track_manager.track_info[track_id]
            track_states[track_id] = {
                "state": track.state.x,
                "covariance": track.state.P,
                "age": info.age,
                "hits": info.hits,
                "last_update_time": info.last_update_time,
            }

        return track_states

    def get_all_track_states(self) -> Dict[int, Dict[str, Any]]:
        """
        Get states of all active tracks (confirmed and tentative).

        Returns:
            Dictionary mapping track_id to track state information
        """
        active_tracks = self.track_manager.get_all_active_tracks()
        track_states = {}

        for track_id, track in active_tracks.items():
            info = self.track_manager.track_info[track_id]
            track_states[track_id] = {
                "state": track.state.x,
                "covariance": track.state.P,
                "age": info.age,
                "hits": info.hits,
                "misses": info.misses,
                "is_confirmed": info.is_confirmed,
                "last_update_time": info.last_update_time,
            }

        return track_states

    def predict_tracks(self, dt: float) -> Dict[int, jnp.ndarray]:
        """
        Predict track states forward in time without updating internal state.

        Args:
            dt: Time step for prediction

        Returns:
            Dictionary mapping track_id to predicted state
        """
        predictions = {}

        for track_id, track in self.track_manager.get_all_active_tracks().items():
            # Predict state without modifying the track
            predicted_state = self.config.motion_model_class.transition(
                track.state.x, dt
            )
            predictions[track_id] = predicted_state

        return predictions

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.track_manager = TrackManager(
            motion_model_class=self.config.motion_model_class,
            confirmation_threshold=self.config.confirmation_threshold,
            deletion_threshold=self.config.deletion_threshold,
            max_tracks=self.config.max_tracks,
        )
        self.current_time = 0.0
        self.last_update_time = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracker statistics."""
        track_summary = self.track_manager.get_track_summary()

        return {
            "current_time": self.current_time,
            "total_tracks": track_summary["total"],
            "confirmed_tracks": track_summary["confirmed"],
            "tentative_tracks": track_summary["tentative"],
            "deleted_tracks": track_summary["deleted"],
            "next_track_id": self.track_manager.next_track_id,
            "config": {
                "data_association": self.config.data_association_algorithm,
                "gate_threshold": self.config.gate_threshold,
                "confirmation_threshold": self.config.confirmation_threshold,
                "deletion_threshold": self.config.deletion_threshold,
                "max_tracks": self.config.max_tracks,
            },
        }
