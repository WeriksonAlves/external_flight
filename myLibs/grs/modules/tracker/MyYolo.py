from ..interfaces import TrackerInterface
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import Tuple, List, Optional
import cv2
import numpy as np
import rospy


def ensure_valid_frame(func):
    """
    Decorator to validate the input frame for YOLO processing.
    Ensures that a valid numpy array is provided.
    """
    def wrapper(self, frame: np.ndarray, *args, **kwargs):
        if frame is None or not isinstance(frame, np.ndarray):
            rospy.logerr("Invalid frame: Must be a non-null numpy array.")
            raise ValueError("Invalid frame provided.")
        return func(self, frame, *args, **kwargs)

    return wrapper


class MyYolo(TrackerInterface):
    """
    YOLO-based processor for detecting and tracking individuals in video
    frames. Uses a pre-trained YOLO model for real-time detection and tracking.
    """

    def __init__(self, yolo_model_path: str, show_logs: bool = False) -> None:
        """
        Initializes the YOLO processor with the specified model.

        :param yolo_model_path: Path to the YOLO model file.
        """
        rospy.loginfo("Initializing YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        self.show_logs = show_logs
        self.track_history = defaultdict(list)
        rospy.loginfo("YOLO model successfully loaded.")

    @ensure_valid_frame
    def detect_people(
        self, frame: np.ndarray, persist: bool = True, verbose: bool = False
    ) -> Tuple[List[Results], np.ndarray]:
        """
        Detects and tracks people in the given frame.

        :param frame: The input video frame (numpy array).
        :param persist: If True, tracking data is persisted across frames.
        :param verbose: If True, detailed detection logs are enabled.
        :return: Tuple containing detection results and an annotated frame.
        """
        if self.show_logs:
            rospy.loginfo("Running YOLO detection on the frame...")
        detection_results = self.yolo_model.track(
            frame, persist=persist, verbose=verbose
        )
        annotated_frame = detection_results[0].plot()
        if self.show_logs:
            rospy.loginfo("YOLO detection completed.")
        return detection_results, annotated_frame

    def identify_operator(
        self, detection_results: List[Results]
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Extracts the bounding box and tracking ID for a detected individual.

        :param detection_results: List of detection results from the YOLO
                                    model.
        :return: Tuple containing the bounding box and tracking ID, or (None,
                    None) if no operator is detected.
        """
        detection_result = detection_results[0].boxes
        if detection_result:
            bounding_boxes = detection_result.xywh.cpu().numpy()
            track_ids = detection_result.id.cpu().numpy().astype(int).tolist()

            for box, track_id in zip(bounding_boxes, track_ids):
                if self.show_logs:
                    rospy.loginfo(
                        f"Operator detected: Track ID = {track_id}, "
                        f"Bounding Box = {box}"
                    )
                return np.array(box, dtype=int), track_id

        rospy.logwarn("No operator detected in the frame.")
        return None, None

    def _draw_track_history(
        self, annotated_frame: np.ndarray, track_id: int,
        bounding_box: np.ndarray, track_length: int
    ) -> None:
        """
        Draws the track history for a specific person on the annotated frame.

        :param annotated_frame: Frame with annotations.
        :param track_id: ID of the tracked person.
        :param bounding_box: Bounding box for the person.
        :param track_length: Maximum length of the tracking path.
        """
        x, y, w, h = bounding_box
        # Add center point to track history
        self.track_history[track_id].append((x + w // 2, y + h // 2))
        self.track_history[track_id] = self.track_history[track_id][
            -track_length:
        ]

        # Draw track history as a polyline
        points = np.array(
            self.track_history[track_id], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(
            annotated_frame, [points], isClosed=False, color=(230, 230, 230),
            thickness=2
        )
        if self.show_logs:
            rospy.loginfo(f"Track history updated for Track ID = {track_id}.")

    def crop_operator(
        self, bounding_box: Optional[np.ndarray], track_id: Optional[int],
        annotated_frame: np.ndarray, frame: np.ndarray, track_length: int = 90
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Highlights the operator, tracks their movements, and crops the ROI.

        :param bounding_box: Bounding box of the detected person.
        :param track_id: Tracking ID of the detected person.
        :param annotated_frame: Frame with annotations.
        :param frame: Original input frame.
        :param track_length: Maximum length of the tracking path.
        :return: Tuple (success, cropped ROI) where success is True if an
                    operator is detected.
        """
        if bounding_box is None or track_id is None:
            rospy.logwarn("Cannot crop operator: No valid detection.")
            return False, None

        x, y, w, h = bounding_box
        # Draw track history on the frame
        self._draw_track_history(
            annotated_frame, track_id, bounding_box, track_length
        )

        # Crop the operator's ROI
        cropped_y1 = max(0, y - h // 2)
        cropped_y2 = y + h // 2
        cropped_x1 = max(0, x - w // 2)
        cropped_x2 = x + w // 2
        operator_roi = frame[cropped_y1:cropped_y2, cropped_x1:cropped_x2]

        # Flip ROI for display consistency
        operator_roi = cv2.flip(operator_roi, 1)
        if self.show_logs:
            rospy.loginfo(f"Cropped operator ROI for Track ID = {track_id}.")
        return True, operator_roi
