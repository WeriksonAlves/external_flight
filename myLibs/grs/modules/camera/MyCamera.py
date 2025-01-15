from ..interfaces.CameraInterface import CameraInterface
from ....rospy_uav.modules.Bebop2 import Bebop2
from typing import Tuple, Union
import cv2
import numpy as np
import rospy
import time


class CameraSetup(CameraInterface):
    """
    Manages video capture from a standard camera or a Bebop2 drone camera.

    This class provides methods to initialize the video source, capture frames,
    and release resources when the video source is no longer needed.
    """

    def __init__(self, source: Union[int, str, Bebop2]) -> None:
        """
        Initialize the camera with the given source.

        :param source: The video source to use (camera index, video file path,
                        or Bebop2 instance).
        :raises RuntimeError: If the camera source cannot be opened.
        """
        self.cap = self._initialize_camera(source)

        if not self.cap.isOpened():
            rospy.logerr("Error: Could not open the camera source.")
            raise RuntimeError("Error: Could not open the camera source.")

        rospy.loginfo(f"Camera initialized successfully with source: {source}")

    def _initialize_camera(self, source: Union[int, str, Bebop2]):
        """
        Private method to initialize the video capture source.

        :param source: The video source (camera index, video file path, or
                        Bebop2 instance).
        :return: A camera object that conforms to the OpenCV interface.
        """
        if isinstance(source, Bebop2):
            rospy.loginfo("Initializing Bebop2 camera adapter.")
            return BebopCameraAdapter(source)
        elif isinstance(source, str):  # Assuming string is a video file path
            rospy.loginfo("Initializing video file adapter.")
            return RecordingVideoAdapter(source)
        rospy.loginfo("Initializing standard OpenCV camera.")
        return cv2.VideoCapture(source)

    def capture_frame(self) -> Union[np.ndarray, None]:
        """
        Capture a single frame from the video source.

        :return: The captured frame as a NumPy array, or None if the capture
                fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture a frame. Retrying...")
            return None

        rospy.logdebug("Frame captured successfully.")
        return frame

    def release(self) -> None:
        """
        Release the video source and clean up resources.
        """
        if hasattr(self.cap, "release"):
            self.cap.release()
            rospy.loginfo("Camera resources released successfully.")

    def __del__(self):
        """
        Ensure resources are released when the object is destroyed.
        """
        self.release()


class BebopCameraAdapter:
    """
    Adapter for integrating the Bebop2 drone camera with a standard interface.

    This class translates the Bebop2 camera API into a format compatible with
    OpenCV's interface.
    """

    def __init__(self, drone: Bebop2, show_logs: bool = False) -> None:
        """
        Initialize the Bebop2 camera adapter.

        :param drone: The Bebop2 drone instance.
        :param show_logs: Whether to log additional details about camera
                            operations (default: False).
        """
        self.drone = drone
        self.show_logs = show_logs
        rospy.loginfo("BebopCameraAdapter initialized.")

    def isOpened(self) -> bool:
        """
        Check if the Bebop2 camera is available.

        :return: True if the camera is open, False otherwise.
        """
        is_open = self.drone.camera_on()
        rospy.logdebug(
            f"Bebop2 camera status: {'Opened' if is_open else 'Closed'}"
        )
        return is_open

    def read(self) -> Tuple[bool, Union[np.ndarray, None]]:
        """
        Capture a single frame from the Bebop2 camera.

        :return: A tuple containing the success status and the captured frame
                    as a NumPy array.
        """
        ret, frame = self.drone.read_image("image")
        if self.show_logs:
            if ret:
                rospy.loginfo(
                    "Frame captured successfully from Bebop2 camera."
                )
            else:
                rospy.logwarn("Failed to capture a frame from Bebop2 camera.")
        return ret, frame

    def release(self) -> None:
        """
        Release the Bebop2 camera resources.
        """
        self.drone.release_camera()
        rospy.loginfo("Bebop2 camera released successfully.")


class RecordingVideoAdapter:
    """
    Adapter for integrating a recorded video file as a video source
    compatible with the standard camera interface.

    This class translates the recorded video file into a format compatible
    with OpenCV's interface.
    """

    def __init__(self, video_path: str, show_logs: bool = False) -> None:
        """
        Initialize the recorded video adapter.

        :param video_path: Path to the recorded video file.
        :param show_logs: Whether to log additional details about video
                          operations (default: False).
        """
        self.video_path = video_path
        self.show_logs = show_logs
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            rospy.logerr(f"Error: Could not open video file at {video_path}.")
            raise RuntimeError(
                f"Error: Could not open video file at {video_path}."
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Retrieve frame rate
        if self.fps <= 0:
            rospy.logwarn("Invalid FPS detected. Defaulting to 30 FPS.")
            self.fps = 30.0  # Fallback to a default value

        self.frame_delay = 1.0 / self.fps  # Calculate frame delay
        self.last_frame_time = time.time()
        rospy.loginfo(
            f"RecordingVideoAdapter initialized with file: "
            f"{video_path}, FPS: {self.fps}."
        )

    def isOpened(self) -> bool:
        """
        Check if the video file is available and ready to read.

        :return: True if the video file is open, False otherwise.
        """
        is_open = self.cap.isOpened()
        if self.show_logs:
            rospy.logdebug(
                f"Recorded video status: {'Opened' if is_open else 'Closed'}"
            )
        return is_open

    def read(self) -> Tuple[bool, Union[np.ndarray, None]]:
        """
        Capture a single frame from the video file.

        Synchronizes the frame reading with the video's frame rate to ensure
        real-time playback.

        :return: A tuple containing the success status and the captured frame
                as a NumPy array.
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time

        if elapsed_time >= self.frame_delay:
            ret, frame = self.cap.read()
            self.last_frame_time = time.time()  # update last frame time
        else:
            ret, frame = False, None

        if self.show_logs:
            if ret:
                rospy.loginfo("Frame captured successfully from video.")
            else:
                rospy.logwarn("Failed to capture a frame from the video.")
        return ret, frame

    def release(self) -> None:
        """
        Release the video file resources.
        """
        if self.cap.isOpened():
            self.cap.release()
            rospy.loginfo("Video file resources released successfully.")
