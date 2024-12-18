from abc import ABC, abstractmethod
from typing import Union


class CameraInterface(ABC):
    """
    Interface for camera or video capture functionality. Defines the essential
    methods for initializing, capturing frames, and releasing camera resources.
    """

    @abstractmethod
    def __init__(self, source: Union[int, str], fps: int = 5,
                 dist: float = 0.025, length: int = 15) -> None:
        """
        Initializes the camera with the specified parameters.

        :param source: Camera source (either an integer index or a string for
                        video source).
        :param fps: Frames per second for video capture (default: 5).
        :param dist: Distance parameter (default: 0.025).
        :param length: Length parameter (default: 15).
        """
        raise NotImplementedError(
            "Subclasses must implement '__init__' method."
        )

    @abstractmethod
    def capture_frame(self) -> Union[None, any]:
        """
        Captures a single frame from the camera.

        :return: The captured frame if successful, None otherwise.
        """
        raise NotImplementedError(
            "Subclasses must implement 'capture_frame' method."
        )

    @abstractmethod
    def get_fps(self) -> int:
        """
        Gets the current frames per second (fps) of the camera.

        :return: FPS value of the camera capture.
        """
        raise NotImplementedError(
            "Subclasses must implement 'get_fps' method."
        )

    @abstractmethod
    def release(self) -> None:
        """
        Releases the camera and cleans up resources when the object is
        destroyed or no longer needed.
        """
        raise NotImplementedError(
            "Subclasses must implement 'release' method."
        )
