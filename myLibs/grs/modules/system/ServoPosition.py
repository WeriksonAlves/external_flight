from ..ros.EspCamServo import EspCamServo
from typing import Tuple
import numpy as np
import rospy


class ServoPosition:
    """
    Handles servo control to adjust the camera's position based on bounding
    box detection.
    """

    def __init__(
        self, num_servos: int = 0, pub_hor_rot: rospy.Publisher = None,
        pub_ver_rot: rospy.Publisher = None, dir_rot: int = 1
    ) -> None:
        """
        Initializes the ServoPositionSystem.

        :param num_servos: Number of servos in the system.
        :param pub_hor_rot: ROS publisher for horizontal rotation commands.
        :param pub_ver_rot: ROS publisher for vertical rotation commands.
        :param dir_rot: Direction multiplier for servo rotation (default is 1).
        """
        self.num_servos = num_servos
        self.enabled = self.num_servos > 0
        self.espcam_servo = EspCamServo(
            pub_hor_rot, pub_ver_rot, dir_rot
        ) if self.enabled else None

        rospy.loginfo(
            "ServoPositionSystem initialized. Enabled: %s", self.enabled
        )

    def adjust_servo_positions(
        self, frame: np.ndarray, bounding_box: Tuple[int, int, int, int]
    ) -> None:
        """
        Adjust servo positions to center the bounding box in the frame.

        :param frame: Captured frame as a numpy array.
        :param bounding_box: Bounding box coordinates (x, y, width, height).
        """
        if not self.enabled:
            rospy.logwarn("Servo system is disabled. Skipping adjustment.")
            return

        frame_center = self._calculate_frame_center(frame)
        box_center = self._calculate_box_center(bounding_box)

        # Calculate distances from the frame center
        horizontal_distance = box_center[0] - frame_center[0]
        self._adjust_servo(horizontal_distance, axis="horizontal")

        if self.num_servos > 1:
            vertical_distance = box_center[1] - frame_center[1]
            self._adjust_servo(vertical_distance, axis="vertical")

    @staticmethod
    def _calculate_frame_center(frame: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the center of the frame.

        :param frame: Captured frame as a numpy array.
        :return: Tuple representing (center_x, center_y) of the frame.
        """
        height, width = frame.shape[:2]
        return width // 2, height // 2

    @staticmethod
    def _calculate_box_center(
        bounding_box: Tuple[int, int, int, int]
    ) -> Tuple[int, int]:
        """
        Calculate the center of the bounding box.

        :param bounding_box: Bounding box coordinates (x, y, width, height).
        :return: Tuple representing (center_x, center_y) of the bounding box.
        """
        x, y, width, height = bounding_box
        return x + width // 2, y + height // 2

    def _adjust_servo(self, distance: int, axis: str) -> None:
        """
        Adjust the servo for the specified axis based on the distance.

        :param distance: Distance from the frame center to the bounding box
                            center.
        :param axis: The axis to adjust ('horizontal' or 'vertical').
        """
        if axis == "horizontal":
            action = "+1" if distance < 0 else "-1"
        elif axis == "vertical":
            action = "+2" if distance < 0 else "-2"
        else:
            rospy.logerr("Invalid axis specified for servo adjustment.")
            return

        rospy.loginfo(
            "Adjusting %s servo with action: %s, Distance: %d",
            axis, action, distance
        )
        self.espcam_servo.perform_action(action, distance)

    def close(self) -> None:
        """
        Finish the ServoPositionSystem.
        """
        if self.enabled:
            self.espcam_servo.close()
            rospy.loginfo("ServoPositionSystem finished.")
        else:
            rospy.logwarn("Servo system is disabled. Skipping finish.")
