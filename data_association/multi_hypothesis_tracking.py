import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type

import jax.numpy as jnp

from data_association.association_gate import cubical_gate, ellipsoidal_gate
from data_association.data_association import DataAssociation
from measurement_models.measurement_model import MeasurementModel


@dataclass
class Hypothesis:
    """
    Represents a single hypothesis in MHT.

    A hypothesis is a specific assignment of measurements to tracks,
    including the possibility of new tracks and false alarms.
    """

    track_assignments: Dict[
        int, Optional[int]
    ]  # track_id -> measurement_id (None = no detection)
    new_tracks: List[int]  # measurement_ids that start new tracks
    false_alarms: Set[int]  # measurement_ids that are false alarms
    log_likelihood: float  # Log-likelihood of this hypothesis
    parent_hypothesis: Optional["Hypothesis"] = None

    def __post_init__(self):
        """Validate hypothesis consistency."""
        # Check that each measurement is assigned to at most one category
        all_measurements = set()

        # Add assigned measurements
        for meas_id in self.track_assignments.values():
            if meas_id is not None:
                if meas_id in all_measurements:
                    raise ValueError(f"Measurement {meas_id} assigned multiple times")
                all_measurements.add(meas_id)

        # Add new track measurements
        for meas_id in self.new_tracks:
            if meas_id in all_measurements:
                raise ValueError(f"Measurement {meas_id} assigned multiple times")
            all_measurements.add(meas_id)

        # Add false alarm measurements
        for meas_id in self.false_alarms:
            if meas_id in all_measurements:
                raise ValueError(f"Measurement {meas_id} assigned multiple times")
            all_measurements.add(meas_id)


class MultiHypothesisTracking(DataAssociation):
    """
    Multi-Hypothesis Tracking (MHT) data association algorithm.

    MHT maintains multiple hypotheses about track-measurement associations
    and prunes unlikely hypotheses over time. This allows for more robust
    tracking in cluttered environments with false alarms and missed detections.
    """

    def __init__(
        self,
        gate_threshold: float = 5.0,
        use_ellipsoidal_gate: bool = True,
        max_hypotheses: int = 100,
        hypothesis_prune_threshold: float = -50.0,
        prob_detection: float = 0.9,
        prob_false_alarm: float = 0.1,
        new_track_threshold: float = 0.1,
        n_scan_pruning: int = 3,
    ):
        """
        Initialize Multi-Hypothesis Tracking.

        Args:
            gate_threshold: Threshold for gating measurements
            use_ellipsoidal_gate: Whether to use ellipsoidal or cubical gating
            max_hypotheses: Maximum number of hypotheses to maintain
            hypothesis_prune_threshold: Log-likelihood threshold for pruning hypotheses
            prob_detection: Probability of detection for existing tracks
            prob_false_alarm: Probability of false alarm per measurement
            new_track_threshold: Threshold for creating new tracks
            n_scan_pruning: Number of scans to look back for pruning
        """
        super().__init__()
        self.gate_threshold = gate_threshold
        self.use_ellipsoidal_gate = use_ellipsoidal_gate
        self.max_hypotheses = max_hypotheses
        self.hypothesis_prune_threshold = hypothesis_prune_threshold
        self.prob_detection = prob_detection
        self.prob_false_alarm = prob_false_alarm
        self.new_track_threshold = new_track_threshold
        self.n_scan_pruning = n_scan_pruning

        # Maintain hypothesis tree
        self.hypotheses: List[Hypothesis] = []
        self.scan_count = 0

    def associate(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks using MHT.

        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples

        Returns:
            Dictionary mapping track_index -> measurement_index (best hypothesis)
        """
        if not track_states and not measurements:
            return {}

        self.scan_count += 1

        # Initialize hypotheses if empty
        if not self.hypotheses:
            self._initialize_hypotheses()

        # Generate new hypotheses for current scan
        new_hypotheses = self._generate_hypotheses(track_states, measurements)

        # Update hypothesis list
        self.hypotheses = new_hypotheses

        # Prune hypotheses
        self._prune_hypotheses()

        # Return best hypothesis association
        return self._get_best_association(track_states, measurements)

    def _initialize_hypotheses(self):
        """Initialize with empty hypothesis."""
        empty_hypothesis = Hypothesis(
            track_assignments={}, new_tracks=[], false_alarms=set(), log_likelihood=0.0
        )
        self.hypotheses = [empty_hypothesis]

    def _generate_hypotheses(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> List[Hypothesis]:
        """Generate all valid hypotheses for current scan."""
        if not measurements:
            # No measurements - all tracks miss detection
            return self._generate_miss_hypotheses(track_states)

        new_hypotheses = []

        for parent_hypothesis in self.hypotheses:
            # Generate child hypotheses from this parent
            child_hypotheses = self._generate_child_hypotheses(
                parent_hypothesis, track_states, measurements
            )
            new_hypotheses.extend(child_hypotheses)

        return new_hypotheses

    def _generate_miss_hypotheses(
        self, track_states: List[jnp.ndarray]
    ) -> List[Hypothesis]:
        """Generate hypotheses when no measurements are available."""
        miss_hypotheses = []

        for parent_hypothesis in self.hypotheses:
            # All tracks miss detection
            track_assignments = {i: None for i in range(len(track_states))}

            miss_likelihood = len(track_states) * math.log(1 - self.prob_detection)

            miss_hypothesis = Hypothesis(
                track_assignments=track_assignments,
                new_tracks=[],
                false_alarms=set(),
                log_likelihood=parent_hypothesis.log_likelihood + miss_likelihood,
                parent_hypothesis=parent_hypothesis,
            )
            miss_hypotheses.append(miss_hypothesis)

        return miss_hypotheses

    def _generate_child_hypotheses(
        self,
        parent_hypothesis: Hypothesis,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> List[Hypothesis]:
        """Generate all valid child hypotheses from a parent hypothesis."""
        # Build gating matrix - which measurements can be associated with which tracks
        gating_matrix = self._build_gating_matrix(track_states, measurements)

        # Generate all valid assignment combinations
        valid_assignments = self._enumerate_assignments(
            len(track_states), len(measurements), gating_matrix
        )

        child_hypotheses = []
        for assignment in valid_assignments:
            try:
                hypothesis = self._create_hypothesis_from_assignment(
                    parent_hypothesis, assignment, track_states, measurements
                )
                child_hypotheses.append(hypothesis)
            except ValueError:
                # Skip invalid hypotheses
                continue

        return child_hypotheses

    def _build_gating_matrix(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> jnp.ndarray:
        """Build binary matrix indicating valid track-measurement associations."""
        n_tracks = len(track_states)
        n_measurements = len(measurements)

        gating_matrix = jnp.zeros((n_tracks, n_measurements), dtype=bool)

        for track_idx, track_state in enumerate(track_states):
            for meas_idx, (measurement, measurement_model, R) in enumerate(
                measurements
            ):
                # Apply gating
                if self.use_ellipsoidal_gate:
                    gate_valid = ellipsoidal_gate(
                        track_state, measurement, measurement_model, self.gate_threshold
                    )
                else:
                    gate_valid = cubical_gate(
                        track_state, measurement, measurement_model, self.gate_threshold
                    )

                gating_matrix = gating_matrix.at[track_idx, meas_idx].set(gate_valid)

        return gating_matrix

    def _enumerate_assignments(
        self, n_tracks: int, n_measurements: int, gating_matrix: jnp.ndarray
    ) -> List[Dict[str, any]]:
        """
        Enumerate all valid assignment possibilities.

        Returns list of assignment dictionaries with keys:
        - 'track_assignments': Dict[int, Optional[int]]
        - 'new_tracks': List[int]
        - 'false_alarms': Set[int]
        """
        assignments = []

        # Limit enumeration to prevent combinatorial explosion
        max_assignments = min(1000, self.max_hypotheses * 10)

        # For each track, it can be: undetected, or assigned to any valid measurement
        track_options = []
        for track_idx in range(n_tracks):
            options = [None]  # Undetected option
            for meas_idx in range(n_measurements):
                if gating_matrix[track_idx, meas_idx]:
                    options.append(meas_idx)
            track_options.append(options)

        # Generate all combinations (with early termination)
        count = 0
        for assignment_tuple in itertools.product(*track_options):
            if count >= max_assignments:
                break

            # Convert to assignment dictionary
            track_assignments = {i: assignment_tuple[i] for i in range(n_tracks)}

            # Determine which measurements are used
            used_measurements = set()
            for meas_id in track_assignments.values():
                if meas_id is not None:
                    used_measurements.add(meas_id)

            # Check for conflicts (multiple tracks assigned to same measurement)
            if len(used_measurements) != len(
                [m for m in track_assignments.values() if m is not None]
            ):
                continue  # Skip conflicting assignments

            # Remaining measurements can be new tracks or false alarms
            unused_measurements = set(range(n_measurements)) - used_measurements

            # For each unused measurement, try both new track and false alarm
            for new_track_subset in self._powerset(unused_measurements):
                false_alarms = unused_measurements - set(new_track_subset)

                assignment = {
                    "track_assignments": track_assignments,
                    "new_tracks": list(new_track_subset),
                    "false_alarms": false_alarms,
                }
                assignments.append(assignment)
                count += 1

                if count >= max_assignments:
                    break

            if count >= max_assignments:
                break

        return assignments

    def _powerset(self, iterable):
        """Generate all subsets of an iterable."""
        s = list(iterable)
        # Limit subset size to prevent explosion
        max_size = min(len(s), 3)
        for size in range(max_size + 1):
            for subset in itertools.combinations(s, size):
                yield subset

    def _create_hypothesis_from_assignment(
        self,
        parent_hypothesis: Hypothesis,
        assignment: Dict[str, any],
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Hypothesis:
        """Create hypothesis from assignment and calculate likelihood."""
        track_assignments = assignment["track_assignments"]
        new_tracks = assignment["new_tracks"]
        false_alarms = assignment["false_alarms"]

        # Calculate log-likelihood
        log_likelihood = parent_hypothesis.log_likelihood

        # Likelihood from track detections/misses
        for track_idx, meas_idx in track_assignments.items():
            if meas_idx is not None:
                # Track detected
                log_likelihood += math.log(self.prob_detection)
                # Add measurement likelihood
                measurement, measurement_model, R = measurements[meas_idx]
                track_state = track_states[track_idx]
                log_likelihood += self._calculate_measurement_likelihood(
                    track_state, measurement, measurement_model, R
                )
            else:
                # Track missed
                log_likelihood += math.log(1 - self.prob_detection)

        # Likelihood from new tracks
        for meas_idx in new_tracks:
            log_likelihood += math.log(self.new_track_threshold)

        # Likelihood from false alarms
        for meas_idx in false_alarms:
            log_likelihood += math.log(self.prob_false_alarm)

        return Hypothesis(
            track_assignments=track_assignments,
            new_tracks=new_tracks,
            false_alarms=false_alarms,
            log_likelihood=log_likelihood,
            parent_hypothesis=parent_hypothesis,
        )

    def _calculate_measurement_likelihood(
        self,
        state: jnp.ndarray,
        measurement: jnp.ndarray,
        measurement_model: Type[MeasurementModel],
        R: jnp.ndarray,
    ) -> float:
        """Calculate log-likelihood of measurement given track state."""
        predicted_measurement = measurement_model.predict_measurement(state)
        innovation = measurement - predicted_measurement

        try:
            # Multivariate Gaussian likelihood
            inv_R = jnp.linalg.inv(R)
            det_R = jnp.linalg.det(R)

            # Log-likelihood = -0.5 * (innovation^T * inv_R * innovation + log(det_R) + k*log(2*pi))
            mahalanobis_dist = innovation.T @ inv_R @ innovation
            k = len(innovation)  # Dimension of measurement

            log_likelihood = -0.5 * (
                mahalanobis_dist + math.log(det_R) + k * math.log(2 * math.pi)
            )

            return float(log_likelihood)
        except:
            # Fallback to simple distance-based likelihood
            distance = jnp.linalg.norm(innovation)
            return -float(distance)

    def _prune_hypotheses(self):
        """Prune unlikely hypotheses to maintain computational tractability."""
        if not self.hypotheses:
            return

        # Sort by likelihood (descending)
        self.hypotheses.sort(key=lambda h: h.log_likelihood, reverse=True)

        # Keep only top hypotheses
        self.hypotheses = self.hypotheses[: self.max_hypotheses]

        # Remove hypotheses below threshold
        self.hypotheses = [
            h
            for h in self.hypotheses
            if h.log_likelihood >= self.hypothesis_prune_threshold
        ]

        # Ensure at least one hypothesis remains
        if not self.hypotheses:
            self._initialize_hypotheses()

    def _get_best_association(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """Get association from best (highest likelihood) hypothesis."""
        if not self.hypotheses:
            return {}

        best_hypothesis = max(self.hypotheses, key=lambda h: h.log_likelihood)

        # Convert to expected format (track_index -> measurement_index)
        associations = {}
        for track_idx, meas_idx in best_hypothesis.track_assignments.items():
            if meas_idx is not None:
                associations[track_idx] = meas_idx

        return associations

    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Get the current best hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.log_likelihood)

    def get_hypothesis_count(self) -> int:
        """Get current number of active hypotheses."""
        return len(self.hypotheses)

    def reset(self):
        """Reset the MHT algorithm."""
        self.hypotheses = []
        self.scan_count = 0
