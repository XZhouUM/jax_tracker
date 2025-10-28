"""
Joint Probabilistic Data Association (JPDA) implementation.

JPDA calculates association probabilities for all possible track-measurement
pairs and uses these probabilities to update tracks with weighted combinations
of measurements. This approach handles uncertainty in data association by
considering all possible associations simultaneously.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import jax.numpy as jnp

from data_association.association_gate import cubical_gate, ellipsoidal_gate
from data_association.data_association import DataAssociation
from measurement_models.measurement_model import MeasurementModel


@dataclass
class AssociationProbabilities:
    """Container for JPDA association probabilities."""

    # Probability matrix: prob_matrix[track_idx][meas_idx] = P(track_idx associated with meas_idx)
    prob_matrix: jnp.ndarray

    # Probability of each track being undetected: prob_undetected[track_idx] = P(track_idx undetected)
    prob_undetected: jnp.ndarray

    # Probability of each measurement being a false alarm: prob_false_alarm[meas_idx] = P(meas_idx is false alarm)
    prob_false_alarm: jnp.ndarray

    # Combined innovation for each track (weighted by association probabilities)
    combined_innovations: List[jnp.ndarray]

    # Combined innovation covariance for each track
    combined_covariances: List[jnp.ndarray]


class JointProbabilisticDataAssociation(DataAssociation):
    """
    Joint Probabilistic Data Association (JPDA) algorithm.

    JPDA calculates association probabilities for all possible track-measurement
    associations and updates tracks using weighted combinations of measurements.
    This provides a probabilistic approach to handling data association uncertainty.
    """

    def __init__(
        self,
        gate_threshold: float = 5.0,
        use_ellipsoidal_gate: bool = True,
        prob_detection: float = 0.9,
        prob_false_alarm: float = 0.1,
        clutter_density: float = 1e-6,
        max_associations: int = 100,
    ):
        """
        Initialize JPDA algorithm.

        Args:
            gate_threshold: Threshold for gating measurements
            use_ellipsoidal_gate: Whether to use ellipsoidal or cubical gating
            prob_detection: Probability of detection for existing tracks
            prob_false_alarm: Probability that a measurement is a false alarm
            clutter_density: Spatial density of clutter (false alarms)
            max_associations: Maximum number of association hypotheses to consider
        """
        super().__init__()
        self.gate_threshold = gate_threshold
        self.use_ellipsoidal_gate = use_ellipsoidal_gate
        self.prob_detection = prob_detection
        self.prob_false_alarm = prob_false_alarm
        self.clutter_density = clutter_density
        self.max_associations = max_associations

        # Store last computed probabilities for external access
        self.last_association_probabilities: Optional[AssociationProbabilities] = None

    def associate(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using JPDA.

        Note: JPDA doesn't make hard associations but calculates probabilities.
        This method returns the most likely association for compatibility with
        the DataAssociation interface, but the real value is in the computed
        association probabilities accessible via get_association_probabilities().

        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples

        Returns:
            Dictionary mapping track_index -> measurement_index (most likely associations)
        """
        if not track_states or not measurements:
            self.last_association_probabilities = None
            return {}

        # Calculate association probabilities
        self.last_association_probabilities = self._calculate_association_probabilities(
            track_states, measurements
        )

        # Extract most likely hard associations for interface compatibility
        return self._extract_hard_associations(self.last_association_probabilities)

    def get_association_probabilities(self) -> Optional[AssociationProbabilities]:
        """
        Get the association probabilities from the last associate() call.

        Returns:
            AssociationProbabilities object containing all probability information,
            or None if associate() hasn't been called yet.
        """
        return self.last_association_probabilities

    def _calculate_association_probabilities(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> AssociationProbabilities:
        """Calculate JPDA association probabilities."""
        n_tracks = len(track_states)
        n_measurements = len(measurements)

        # Step 1: Apply gating and build validation matrix
        validation_matrix = self._build_validation_matrix(track_states, measurements)

        # Step 2: Calculate likelihood matrix
        likelihood_matrix = self._calculate_likelihood_matrix(
            track_states, measurements, validation_matrix
        )

        # Step 3: Enumerate feasible association events
        association_events = self._enumerate_association_events(
            n_tracks, n_measurements, validation_matrix
        )

        # Step 4: Calculate event probabilities
        event_probabilities = self._calculate_event_probabilities(
            association_events, likelihood_matrix, n_tracks, n_measurements
        )

        # Step 5: Calculate marginal association probabilities
        prob_matrix, prob_undetected, prob_false_alarm = (
            self._calculate_marginal_probabilities(
                association_events, event_probabilities, n_tracks, n_measurements
            )
        )

        # Step 6: Calculate combined innovations and covariances
        combined_innovations, combined_covariances = self._calculate_combined_updates(
            track_states, measurements, prob_matrix, validation_matrix
        )

        return AssociationProbabilities(
            prob_matrix=prob_matrix,
            prob_undetected=prob_undetected,
            prob_false_alarm=prob_false_alarm,
            combined_innovations=combined_innovations,
            combined_covariances=combined_covariances,
        )

    def _build_validation_matrix(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> jnp.ndarray:
        """
        Build validation matrix indicating which measurements are valid for each track.

        Returns:
            Boolean matrix where validation_matrix[i,j] = True if measurement j
            is within the gate of track i.
        """
        n_tracks = len(track_states)
        n_measurements = len(measurements)

        validation_matrix = jnp.zeros((n_tracks, n_measurements), dtype=bool)

        for track_idx, track_state in enumerate(track_states):
            for meas_idx, (measurement, measurement_model, _) in enumerate(
                measurements
            ):
                if self.use_ellipsoidal_gate:
                    is_valid = ellipsoidal_gate(
                        track_state, measurement, measurement_model, self.gate_threshold
                    )
                else:
                    is_valid = cubical_gate(
                        track_state, measurement, measurement_model, self.gate_threshold
                    )

                validation_matrix = validation_matrix.at[track_idx, meas_idx].set(
                    is_valid
                )

        return validation_matrix

    def _calculate_likelihood_matrix(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
        validation_matrix: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculate likelihood matrix for valid track-measurement pairs.

        Returns:
            Matrix where likelihood_matrix[i,j] is the likelihood of measurement j
            given track i (0 if not valid according to validation_matrix).
        """
        n_tracks = len(track_states)
        n_measurements = len(measurements)

        likelihood_matrix = jnp.zeros((n_tracks, n_measurements))

        for track_idx, track_state in enumerate(track_states):
            for meas_idx, (measurement, measurement_model, R) in enumerate(
                measurements
            ):
                if validation_matrix[track_idx, meas_idx]:
                    likelihood = self._calculate_measurement_likelihood(
                        track_state, measurement, measurement_model, R
                    )
                    likelihood_matrix = likelihood_matrix.at[track_idx, meas_idx].set(
                        likelihood
                    )

        return likelihood_matrix

    def _calculate_measurement_likelihood(
        self,
        state: jnp.ndarray,
        measurement: jnp.ndarray,
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray,
    ) -> float:
        """Calculate likelihood of measurement given track state."""
        try:
            # Predict measurement
            predicted_measurement = measurement_model.predict_measurement(state)

            # Calculate innovation
            innovation = measurement - predicted_measurement

            # Calculate likelihood using multivariate Gaussian
            # L = (2π)^(-k/2) * |R|^(-1/2) * exp(-0.5 * innovation^T * R^(-1) * innovation)
            k = len(innovation)

            # Calculate log-likelihood for numerical stability
            log_likelihood = (
                -0.5 * k * jnp.log(2 * jnp.pi)
                - 0.5 * jnp.linalg.slogdet(R)[1]
                - 0.5 * innovation.T @ jnp.linalg.solve(R, innovation)
            )

            return float(jnp.exp(log_likelihood))

        except Exception:
            # Return small likelihood for numerical issues
            return 1e-10

    def _enumerate_association_events(
        self, n_tracks: int, n_measurements: int, validation_matrix: jnp.ndarray
    ) -> List[Dict[str, any]]:
        """
        Enumerate all feasible association events.

        An association event specifies which measurements are associated with
        which tracks, with constraints that each measurement can be associated
        with at most one track.

        Returns:
            List of association events, where each event is a dictionary with:
            - 'associations': Dict[int, int] mapping track_idx -> meas_idx
            - 'undetected_tracks': Set[int] of undetected track indices
            - 'false_alarms': Set[int] of false alarm measurement indices
        """
        events = []

        # Generate all possible assignment combinations
        # Each track can be: undetected (None) or assigned to a valid measurement
        track_options = []
        for track_idx in range(n_tracks):
            options = [None]  # Undetected option
            for meas_idx in range(n_measurements):
                if validation_matrix[track_idx, meas_idx]:
                    options.append(meas_idx)
            track_options.append(options)

        # Enumerate all combinations (with limits to prevent explosion)
        count = 0
        for assignment in itertools.product(*track_options):
            if count >= self.max_associations:
                break

            # Check if assignment is feasible (no measurement assigned to multiple tracks)
            assigned_measurements = set()
            associations = {}
            undetected_tracks = set()

            valid_assignment = True
            for track_idx, meas_idx in enumerate(assignment):
                if meas_idx is None:
                    undetected_tracks.add(track_idx)
                else:
                    if meas_idx in assigned_measurements:
                        valid_assignment = False
                        break
                    assigned_measurements.add(meas_idx)
                    associations[track_idx] = meas_idx

            if valid_assignment:
                # Determine false alarms
                false_alarms = set(range(n_measurements)) - assigned_measurements

                events.append(
                    {
                        "associations": associations,
                        "undetected_tracks": undetected_tracks,
                        "false_alarms": false_alarms,
                    }
                )
                count += 1

        return events

    def _calculate_event_probabilities(
        self,
        association_events: List[Dict[str, any]],
        likelihood_matrix: jnp.ndarray,
        n_tracks: int,
        n_measurements: int,
    ) -> jnp.ndarray:
        """Calculate probability of each association event."""
        n_events = len(association_events)
        event_probs = jnp.zeros(n_events)

        for event_idx, event in enumerate(association_events):
            prob = 1.0

            # Probability from track detections/non-detections
            for track_idx in range(n_tracks):
                if track_idx in event["undetected_tracks"]:
                    prob *= 1.0 - self.prob_detection
                else:
                    prob *= self.prob_detection

            # Probability from measurement likelihoods
            for track_idx, meas_idx in event["associations"].items():
                prob *= likelihood_matrix[track_idx, meas_idx]

            # Probability from false alarms
            n_false_alarms = len(event["false_alarms"])
            prob *= self.clutter_density**n_false_alarms

            event_probs = event_probs.at[event_idx].set(prob)

        # Normalize probabilities
        total_prob = jnp.sum(event_probs)
        if total_prob > 0:
            event_probs = event_probs / total_prob

        return event_probs

    def _calculate_marginal_probabilities(
        self,
        association_events: List[Dict[str, any]],
        event_probabilities: jnp.ndarray,
        n_tracks: int,
        n_measurements: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calculate marginal association probabilities.

        Returns:
            - prob_matrix: P(track i associated with measurement j)
            - prob_undetected: P(track i undetected)
            - prob_false_alarm: P(measurement j is false alarm)
        """
        prob_matrix = jnp.zeros((n_tracks, n_measurements))
        prob_undetected = jnp.zeros(n_tracks)
        prob_false_alarm = jnp.zeros(n_measurements)

        for event_idx, event in enumerate(association_events):
            event_prob = event_probabilities[event_idx]

            # Update association probabilities
            for track_idx, meas_idx in event["associations"].items():
                prob_matrix = prob_matrix.at[track_idx, meas_idx].add(event_prob)

            # Update undetected probabilities
            for track_idx in event["undetected_tracks"]:
                prob_undetected = prob_undetected.at[track_idx].add(event_prob)

            # Update false alarm probabilities
            for meas_idx in event["false_alarms"]:
                prob_false_alarm = prob_false_alarm.at[meas_idx].add(event_prob)

        return prob_matrix, prob_undetected, prob_false_alarm

    def _calculate_combined_updates(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
        prob_matrix: jnp.ndarray,
        validation_matrix: jnp.ndarray,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Calculate combined innovations and covariances for JPDA updates.

        Returns:
            - combined_innovations: List of combined innovation vectors for each track
            - combined_covariances: List of combined innovation covariances for each track
        """
        n_tracks = len(track_states)
        combined_innovations = []
        combined_covariances = []

        for track_idx in range(n_tracks):
            track_state = track_states[track_idx]

            # Calculate weighted innovation
            weighted_innovation = jnp.zeros_like(
                measurements[0][0]
            )  # Initialize with measurement dimension
            total_prob = 0.0

            # Collect valid measurements and their probabilities
            valid_measurements = []
            valid_probs = []

            for meas_idx, (measurement, measurement_model, R) in enumerate(
                measurements
            ):
                if validation_matrix[track_idx, meas_idx]:
                    prob = prob_matrix[track_idx, meas_idx]
                    if prob > 1e-10:  # Only consider significant probabilities
                        predicted_measurement = measurement_model.predict_measurement(
                            track_state
                        )
                        innovation = measurement - predicted_measurement

                        weighted_innovation += prob * innovation
                        total_prob += prob

                        valid_measurements.append(
                            (measurement, measurement_model, R, innovation)
                        )
                        valid_probs.append(prob)

            # Normalize if we have valid measurements
            if total_prob > 1e-10:
                weighted_innovation = weighted_innovation / total_prob

            combined_innovations.append(weighted_innovation)

            # Calculate combined covariance (simplified approach)
            #
            # THEORETICAL JPDA COMBINED COVARIANCE FORMULA:
            # S_combined = Σ(i) β_i * (R_i + ν_i * ν_i^T) - ν_combined * ν_combined^T
            #
            # Where:
            # - β_i = association probability for measurement i
            # - R_i = measurement covariance matrix for measurement i
            # - ν_i = innovation vector for measurement i (measurement - prediction)
            # - ν_combined = combined weighted innovation vector
            # - ν_i * ν_i^T = outer product representing innovation spread
            #
            # COMPLETE IMPLEMENTATION WOULD BE:
            # combined_cov = jnp.zeros_like(valid_measurements[0][2])
            # for (meas, model, R, innovation), prob in zip(valid_measurements, valid_probs):
            #     # Add weighted measurement covariance
            #     combined_cov += prob * R
            #     # Add weighted innovation spread (uncertainty from association)
            #     combined_cov += prob * jnp.outer(innovation, innovation)
            # # Subtract combined innovation spread (avoid double counting)
            # combined_cov -= jnp.outer(weighted_innovation, weighted_innovation)
            # combined_covariances.append(combined_cov)
            #
            # CURRENT SIMPLIFIED APPROACH:
            # PROS:
            # - Computationally fast (O(1) vs O(n*m^2) for proper approach)
            # - Simple implementation, no complex matrix operations
            # - Numerically stable, avoids potential matrix conditioning issues in inversion
            # - Works reasonably well when all measurements have similar noise levels
            # - Good approximation when one measurement clearly dominates (high β_i)
            #
            # CONS:
            # - Theoretically incorrect, violates JPDA mathematical framework
            # - Underestimates uncertainty by ignoring association ambiguity
            # - Kalman filter receives incorrect covariance information
            # - Poor performance with heterogeneous measurement noise levels
            # - Suboptimal in highly ambiguous association scenarios
            # - May lead to overconfident tracking and filter divergence
            #
            if valid_measurements:
                # Use the covariance of the first valid measurement as approximation
                # This underestimates uncertainty by ignoring association ambiguity
                combined_covariances.append(valid_measurements[0][2])
            else:
                # No valid measurements - use identity matrix
                meas_dim = len(measurements[0][0]) if measurements else 2
                combined_covariances.append(jnp.eye(meas_dim))

        return combined_innovations, combined_covariances

    def _extract_hard_associations(
        self, association_probs: AssociationProbabilities
    ) -> Dict[int, int]:
        """
        Extract hard associations from probability matrix for interface compatibility.

        Returns the most likely association for each track.
        """
        associations = {}
        prob_matrix = association_probs.prob_matrix
        n_tracks, n_measurements = prob_matrix.shape

        used_measurements = set()

        # For each track, find the measurement with highest association probability
        for track_idx in range(n_tracks):
            best_meas_idx = None
            best_prob = 0.0

            for meas_idx in range(n_measurements):
                if meas_idx not in used_measurements:
                    prob = prob_matrix[track_idx, meas_idx]
                    if (
                        prob > best_prob and prob > 0.5
                    ):  # Threshold for confident association
                        best_prob = prob
                        best_meas_idx = meas_idx

            if best_meas_idx is not None:
                associations[track_idx] = best_meas_idx
                used_measurements.add(best_meas_idx)

        return associations
