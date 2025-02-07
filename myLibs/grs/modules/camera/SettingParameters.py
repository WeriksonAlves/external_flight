import rospy


class SingletonMeta(type):
    """
    A Singleton metaclass for ensuring only one instance of a class is created.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SettingParameters(metaclass=SingletonMeta):
    """
    A Singleton class to manage camera settings and gesture recognition
    parameters.
    """

    def __init__(
        self, fps: int = 5, dist: float = 0.025, length: int = 15
    ) -> None:
        """
        Initialize the SettingParameters class with default or provided
        configuration values.

        :param fps: Frames per second for video capture (default: 5).
        :param dist: Distance threshold for gesture recognition (default:
                        0.025).
        :param length: Length parameter used in gesture tracking (default: 15).
        """
        if hasattr(self, "_initialized") and self._initialized:
            rospy.logwarn(
                "SettingParameters instance already initialized. Skipping "
                "reinitialization."
            )
            return

        self.fps = fps
        self.dist = dist
        self.length = length

        rospy.loginfo(
            f"SettingParameters initialized with fps={fps}, dist={dist}, "
            f"length={length}"
        )

        self._initialized = True

    def get_fps(self) -> int:
        """
        Get the current frames per second (fps) setting.

        :return: Frames per second value.
        """
        rospy.logdebug(f"Fetching FPS: {self.fps}")
        return self.fps

    def set_fps(self, fps: int) -> None:
        """
        Set a new frames per second (fps) value.

        :param fps: New frames per second value.
        """
        if fps <= 0:
            rospy.logwarn("FPS value must be greater than 0. No changes made.")
            return

        rospy.loginfo(f"Updating FPS from {self.fps} to {fps}")
        self.fps = fps

    def get_distance(self) -> float:
        """
        Get the current distance threshold.

        :return: Distance threshold value.
        """
        rospy.logdebug(f"Fetching Distance: {self.dist}")
        return self.dist

    def set_distance(self, dist: float) -> None:
        """
        Set a new distance threshold value.

        :param dist: New distance threshold value.
        """
        if dist <= 0:
            rospy.logwarn(
                "Distance value must be greater than 0. No changes made."
            )
            return

        rospy.loginfo(f"Updating Distance from {self.dist} to {dist}")
        self.dist = dist

    def get_length(self) -> int:
        """
        Get the current length parameter value.

        :return: Length parameter value.
        """
        rospy.logdebug(f"Fetching Length: {self.length}")
        return self.length

    def set_length(self, length: int) -> None:
        """
        Set a new length parameter value.

        :param length: New length parameter value.
        """
        if length <= 0:
            rospy.logwarn(
                "Length value must be greater than 0. No changes made."
            )
            return

        rospy.loginfo(f"Updating Length from {self.length} to {length}")
        self.length = length
