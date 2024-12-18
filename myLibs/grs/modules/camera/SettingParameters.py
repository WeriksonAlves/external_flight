import rospy


class SettingParameters:
    """
    Class to manage camera settings and gesture recognition parameters.

    Attributes:
        fps (int): Frames per second for video capture (default: 5).
        dist (float): Distance parameter for gesture recognition (default:
                        0.025).
        length (int): Length parameter used in tracking (default: 15).
    """

    def __init__(
        self, fps: int = 5, dist: float = 0.025, length: int = 15
    ) -> None:
        """
        Initializes the SettingParameters class with the provided
        configuration values.

        Args:
            fps (int): Frames per second for video capture. Default is 5.
            dist (float): Distance threshold for gesture recognition. Default
                            is 0.025.
            length (int): Length parameter for tracking. Default is 15.
        """
        self.fps = fps
        self.dist = dist
        self.length = length

        rospy.loginfo(
            f"SettingParameters initialized with "
            f"fps={fps}, dist={dist}, length={length}"
        )

    def get_fps(self) -> int:
        """
        Returns the current frames per second (fps) setting.

        Returns:
            int: Frames per second for video capture.
        """
        rospy.loginfo(f"Fetching FPS: {self.fps}")
        return self.fps

    def set_fps(self, fps: int) -> None:
        """
        Updates the frames per second (fps) setting.

        Args:
            fps (int): New frames per second value.
        """
        if fps <= 0:
            rospy.logwarn("FPS value must be greater than 0. No changes made.")
            return

        rospy.loginfo(f"Updating FPS from {self.fps} to {fps}")
        self.fps = fps

    def get_distance(self) -> float:
        """
        Returns the distance threshold setting.

        Returns:
            float: Distance threshold for gesture recognition.
        """
        rospy.loginfo(f"Fetching Distance: {self.dist}")
        return self.dist

    def set_distance(self, dist: float) -> None:
        """
        Updates the distance threshold setting.

        Args:
            dist (float): New distance threshold value.
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
        Returns the length parameter setting.

        Returns:
            int: Length parameter used in gesture tracking.
        """
        rospy.loginfo(f"Fetching Length: {self.length}")
        return self.length

    def set_length(self, length: int) -> None:
        """
        Updates the length parameter setting.

        Args:
            length (int): New length parameter value.
        """
        if length <= 0:
            rospy.logwarn(
                "Length value must be greater than 0. No changes made."
            )
            return

        rospy.loginfo(f"Updating Length from {self.length} to {length}")
        self.length = length
