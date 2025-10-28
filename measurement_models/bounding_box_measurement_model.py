import jax.numpy as jnp

from measurement_models.measurement_model import MeasurementModel


class BoundingBoxMeasurement(MeasurementModel):
    """
    Bounding box measurement model for object detection tracking.

    This model handles bounding box measurements from object detectors.
    The state represents the bounding box center and size with velocities.

    State format: [cx, cy, w, h, vcx, vcy, vw, vh]
    - cx, cy: center coordinates
    - w, h: width and height
    - vcx, vcy: velocity of center
    - vw, vh: velocity of size (rate of change)

    Measurement format: [cx, cy, w, h]
    - cx, cy: center coordinates
    - w, h: width and height
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def predict_measurement(state: jnp.ndarray) -> jnp.ndarray:
        """
        Predict measurement from state.

        Args:
            state: State vector [cx, cy, w, h, vcx, vcy, vw, vh]

        Returns:
            Predicted measurement [cx, cy, w, h]
        """
        # Extract position and size components (ignore velocities)
        return state[:4]

    @staticmethod
    def jacobian(state: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobian of measurement function wrt state.

        The measurement function is h(x) = [cx, cy, w, h] = x[:4]
        So the Jacobian is a 4x8 matrix with identity in the first 4x4 block.

        Args:
            state: State vector [cx, cy, w, h, vcx, vcy, vw, vh]

        Returns:
            Jacobian matrix (4x8)
        """
        # Measurement is just the first 4 components of state
        H = jnp.zeros((4, 8))
        H = H.at[:4, :4].set(jnp.eye(4))
        return H

    @staticmethod
    def state_to_bbox(state: jnp.ndarray) -> jnp.ndarray:
        """
        Convert state to bounding box format [x1, y1, x2, y2].

        Args:
            state: State vector [cx, cy, w, h, vcx, vcy, vw, vh]

        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        cx, cy, w, h = state[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return jnp.array([x1, y1, x2, y2])

    @staticmethod
    def measurement_to_bbox(measurement: jnp.ndarray) -> jnp.ndarray:
        """
        Convert measurement to bounding box format [x1, y1, x2, y2].

        Args:
            measurement: Measurement vector [cx, cy, w, h]

        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        cx, cy, w, h = measurement
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return jnp.array([x1, y1, x2, y2])

    @staticmethod
    def bbox_to_measurement(bbox: jnp.ndarray) -> jnp.ndarray:
        """
        Convert bounding box [x1, y1, x2, y2] to measurement format.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Measurement [cx, cy, w, h]
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return jnp.array([cx, cy, w, h])

    @staticmethod
    def calculate_iou(bbox1: jnp.ndarray, bbox2: jnp.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1_inter = jnp.maximum(bbox1[0], bbox2[0])
        y1_inter = jnp.maximum(bbox1[1], bbox2[1])
        x2_inter = jnp.minimum(bbox1[2], bbox2[2])
        y2_inter = jnp.minimum(bbox1[3], bbox2[3])

        # Calculate intersection area
        inter_width = jnp.maximum(0.0, x2_inter - x1_inter)
        inter_height = jnp.maximum(0.0, y2_inter - y1_inter)
        intersection = inter_width * inter_height

        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        # Avoid division by zero
        iou = jnp.where(union > 1e-8, intersection / union, 0.0)
        return iou
