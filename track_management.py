from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import jax.numpy as jnp

from kalman_filter_track import KalmanFilterTrack, KalmanFilterTrackState
from measurement_models.measurement_model import MeasurementModel
from motion_models.motion_model import MotionModel


@dataclass
class TrackInfo:
    """Information about a track's lifecycle and status."""

    track_id: int
    age: int  # Number of time steps since track creation
    hits: int  # Number of successful measurement associations
    misses: int  # Number of consecutive missed detections
    created_time: float
    last_update_time: float
    is_confirmed: bool = False
    is_deleted: bool = False


class TrackManager:
    """
    Manages track lifecycle including initialization, confirmation, and deletion.
    """

    def __init__(
        self,
        motion_model_class: Type[MotionModel],
        confirmation_threshold: int = 3,
        deletion_threshold: int = 5,
        max_tracks: int = 100,
    ):
        """
        Initialize track manager.

        Args:
            motion_model_class: Motion model class for new tracks
            confirmation_threshold: Number of hits needed to confirm a track
            deletion_threshold: Number of consecutive misses before deletion
            max_tracks: Maximum number of tracks to maintain
        """
        self.motion_model_class = motion_model_class
        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        self.max_tracks = max_tracks

        self.tracks: Dict[int, KalmanFilterTrack] = {}
        self.track_info: Dict[int, TrackInfo] = {}
        self.next_track_id = 0

    def initialize_track(
        self,
        measurement: jnp.ndarray,
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray,
        current_time: float,
        initial_covariance_scale: float = 10.0,
    ) -> int:
        """
        Initialize a new track from a measurement.

        Args:
            measurement: Initial measurement
            measurement_model: Measurement model class
            R: Measurement noise covariance
            current_time: Current time
            initial_covariance_scale: Scale factor for initial state covariance

        Returns:
            Track ID of the new track
        """
        if len(self.tracks) >= self.max_tracks:
            # Remove oldest unconfirmed track to make space
            self._remove_oldest_unconfirmed_track()

        # Initialize state from measurement
        initial_state = self._initialize_state_from_measurement(
            measurement, measurement_model, R, initial_covariance_scale
        )

        # Create track
        track_id = self.next_track_id
        self.next_track_id += 1

        track = KalmanFilterTrack(initial_state, self.motion_model_class)
        self.tracks[track_id] = track

        # Create track info
        self.track_info[track_id] = TrackInfo(
            track_id=track_id,
            age=0,
            hits=1,  # Start with 1 hit from initialization
            misses=0,
            created_time=current_time,
            last_update_time=current_time,
        )

        return track_id

    def update_track_status(
        self, track_id: int, associated: bool, current_time: float
    ) -> None:
        """
        Update track status based on association result.

        Args:
            track_id: ID of the track to update
            associated: Whether the track was associated with a measurement
            current_time: Current time
        """
        if track_id not in self.track_info:
            return

        info = self.track_info[track_id]
        info.age += 1
        info.last_update_time = current_time

        if associated:
            info.hits += 1
            info.misses = 0  # Reset miss counter

            # Check for confirmation
            if not info.is_confirmed and info.hits >= self.confirmation_threshold:
                info.is_confirmed = True
        else:
            info.misses += 1

            # Check for deletion
            if info.misses >= self.deletion_threshold:
                info.is_deleted = True

    def get_confirmed_tracks(self) -> Dict[int, KalmanFilterTrack]:
        """Get all confirmed tracks."""
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if self.track_info[track_id].is_confirmed
            and not self.track_info[track_id].is_deleted
        }

    def get_all_active_tracks(self) -> Dict[int, KalmanFilterTrack]:
        """Get all active (non-deleted) tracks."""
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if not self.track_info[track_id].is_deleted
        }

    def cleanup_deleted_tracks(self) -> List[int]:
        """
        Remove deleted tracks from memory.

        Returns:
            List of deleted track IDs
        """
        deleted_ids = [
            track_id for track_id, info in self.track_info.items() if info.is_deleted
        ]

        for track_id in deleted_ids:
            del self.tracks[track_id]
            del self.track_info[track_id]

        return deleted_ids

    def get_track_states(
        self, confirmed_only: bool = False
    ) -> List[Tuple[int, jnp.ndarray]]:
        """
        Get track states for data association.

        Args:
            confirmed_only: Whether to return only confirmed tracks

        Returns:
            List of (track_id, state_vector) tuples
        """
        if confirmed_only:
            tracks = self.get_confirmed_tracks()
        else:
            tracks = self.get_all_active_tracks()

        return [(track_id, track.state.x) for track_id, track in tracks.items()]

    def _initialize_state_from_measurement(
        self,
        measurement: jnp.ndarray,
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray,
        covariance_scale: float,
    ) -> KalmanFilterTrackState:
        """
        Initialize track state from first measurement.

        This is a simplified initialization. In practice, you might need
        more sophisticated methods depending on your measurement model.
        """
        # For constant velocity model with position measurements
        # Assume measurement is [x, y] and initialize velocity to zero
        if measurement.shape[0] == 2:  # Position measurement
            initial_x = jnp.array([measurement[0], measurement[1], 0.0, 0.0])
            initial_P = (
                jnp.diag(jnp.array([R[0, 0], R[1, 1], 100.0, 100.0])) * covariance_scale
            )
        elif measurement.shape[0] == 3:  # Range, range-rate, azimuth (radar)
            # Convert polar to Cartesian
            range_val, range_rate, azimuth = measurement
            x = range_val * jnp.cos(azimuth)
            y = range_val * jnp.sin(azimuth)
            vx = range_rate * jnp.cos(azimuth)
            vy = range_rate * jnp.sin(azimuth)

            initial_x = jnp.array([x, y, vx, vy])
            initial_P = jnp.eye(4) * covariance_scale
        else:
            # Generic initialization
            state_dim = 4  # Assume 2D position + velocity
            initial_x = jnp.zeros(state_dim)
            initial_x = initial_x.at[: measurement.shape[0]].set(measurement)
            initial_P = jnp.eye(state_dim) * covariance_scale

        return KalmanFilterTrackState.create(initial_x, initial_P)

    def _remove_oldest_unconfirmed_track(self) -> None:
        """Remove the oldest unconfirmed track to make space for new tracks."""
        unconfirmed_tracks = [
            (track_id, info)
            for track_id, info in self.track_info.items()
            if not info.is_confirmed and not info.is_deleted
        ]

        if unconfirmed_tracks:
            # Find oldest unconfirmed track
            oldest_id = min(unconfirmed_tracks, key=lambda x: x[1].created_time)[0]
            self.track_info[oldest_id].is_deleted = True

    def get_track_summary(self) -> Dict[str, int]:
        """Get summary statistics of tracks."""
        total = len(self.tracks)
        confirmed = sum(
            1
            for info in self.track_info.values()
            if info.is_confirmed and not info.is_deleted
        )
        tentative = sum(
            1
            for info in self.track_info.values()
            if not info.is_confirmed and not info.is_deleted
        )
        deleted = sum(1 for info in self.track_info.values() if info.is_deleted)

        return {
            "total": total,
            "confirmed": confirmed,
            "tentative": tentative,
            "deleted": deleted,
        }
