import cv2
import rospy
from ..interfaces.CameraInterface import CameraInterface
from typing import Union


class StandardCameras(CameraInterface):
    """
    A class that manages the camera or drone-based video capture.

    This class handles initialization of the video capture, including
    connecting to the camera and providing methods for releasing it.
    """

    def __init__(self, source: Union[int, str], fps: int = 5,
                 dist: float = 0.025, length: int = 15) -> None:
        """
        Initializes the camera with the given configuration.

        :param source: Camera source (either an integer index or a string for
                        video source).
        :param fps: Frames per second for video capture (default: 5).
        :param dist: Distance parameter (default: 0.025).
        :param length: Length parameter (default: 15).
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            rospy.logerr("Error: Could not open camera.")
            raise RuntimeError("Error: Could not open camera.")
        rospy.loginfo(f"Camera initialized successfully with source: {source}")

        self.fps = fps
        self.dist = dist
        self.length = length

    def capture_frame(self) -> Union[None, any]:
        """
        Captures a single frame from the camera.

        :return: The captured frame if successful, None otherwise.
        """
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Warning: Failed to capture frame.")
            return None
        return frame

    def get_fps(self) -> int:
        """
        Gets the current frames per second (fps) of the camera.

        :return: FPS value of the camera capture.
        """
        return self.fps

    def release(self) -> None:
        """Releases the camera and cleans up resources."""
        if hasattr(self.cap, 'release'):
            self.cap.release()
            rospy.loginfo("Camera released successfully.")
