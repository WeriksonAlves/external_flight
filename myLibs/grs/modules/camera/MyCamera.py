import cv2
import rospy
import numpy as np
from ..interfaces.CameraInterface import CameraInterface
from ....rospy_uav.modules.Bebop2 import Bebop2
from typing import Union, Tuple


class CameraSetup(CameraInterface):
    """
    A class that manages the camera or drone-based video capture.

    This class handles initialization of the video capture, including
    connecting to the camera and providing methods for releasing it.
    """

    def __init__(self, source: Union[int, str, BebopCameraSetup]) -> None:
        """
        Initializes the camera with the given configuration.

        :param source: Camera source (either an integer index or a string for
                        video source).
        :param fps: Frames per second for video capture (default: 5).
        :param dist: Distance parameter (default: 0.025).
        :param length: Length parameter (default: 15).
        """
        if hasattr(source, 'drone_type'):
            self.cap = BebopCameraSetup  # ROS-based camera
        else:
            self.cap = cv2.VideoCapture(source)  # OpenCV-based camera

        if not self.cap.isOpened():
            rospy.logerr("Error: Could not open camera.")
            raise RuntimeError("Error: Could not open camera.")
        rospy.loginfo(f"Camera initialized successfully with source: {source}")

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

    def release(self) -> None:
        """Releases the camera and cleans up resources."""
        if hasattr(self.cap, 'release'):
            self.cap.release()
            rospy.loginfo("Camera released successfully.")

    def __del__(self):
        """ Release the camera when object is destroyed. """
        if hasattr(self.cap, 'release'):
            self.cap.release()


class BebopCameraSetup():
    def __init__(self, drone: Bebop2) -> None:
        self.drone = drone

    def isOpened(self) -> bool:
        if self.drone.camera_on():
            return True
        return False

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.drone.read_image()
        return ret, frame if ret else None, None

    def release(self) -> None:
        """Releases the camera and cleans up resources."""
        self.drone.release_camera()
