import rospy
import numpy as np
from std_msgs.msg import Int32
from typing import Dict, List


class EspCamServo:
    """
    Manages servo communication for camera rotation adjustments.
    """

    def __init__(
        self, horizontal_publisher: rospy.Publisher,
        vertical_publisher: rospy.Publisher, dir_rot: int = 1
    ) -> None:
        """
        Initializes the CommunicationEspCam object.

        :param horizontal_publisher: ROS publisher for horizontal servo
                                        control.
        :param vertical_publisher: ROS publisher for vertical servo control.
        :param dir_rot: Direction multiplier for servo rotation (default is 1).
        """
        self.horizontal_publisher = horizontal_publisher
        self.vertical_publisher = vertical_publisher
        self.dir_rot = dir_rot
        rospy.loginfo(
            "CommunicationEspCam initialized with dir_rot=%d.", dir_rot
        )

    def perform_action(
        self, action: str, distance_to_center: int, show_message: bool = False,
        gains: List = [5, 1 / 30]
    ) -> None:
        """
        Sends control commands to the servos based on the specified action.

        :param action: Action command for the servo ("+1", "-1", "+2", "-2").
        :param distance_to_center: Distance to the center for servo
                                    adjustments.
        :param show_message: Flag to indicate whether to log the action
                                message (default is False).
        """
        action_messages: Dict[str, str] = {
            '+1': "Turn the horizontal servo to the left.",
            '-1': "Turn the horizontal servo to the right.",
            '+2': "Turn the vertical servo upwards.",
            '-2': "Turn the vertical servo downwards.",
        }

        # Calculate the servo control signal
        signal = round(
            self.dir_rot * gains[0] * np.tanh(gains[1] * distance_to_center)
        )

        if show_message:
            rospy.loginfo(
                "Action: %s | Distance: %d | Signal: %d | Message: %s",
                action, distance_to_center, signal, action_messages.get(
                    action, "Invalid action."
                )
            )

        # Publish the control command
        if action in ['+1', '-1']:
            self._publish_signal(
                self.horizontal_publisher, signal, "horizontal"
            )
        elif action in ['+2', '-2']:
            self._publish_signal(
                self.vertical_publisher, signal, "vertical"
            )
        else:
            rospy.logerr("Invalid action command: %s", action)

    @staticmethod
    def _publish_signal(
        publisher: rospy.Publisher, signal: int, axis: str
    ) -> None:
        """
        Helper method to publish the servo control signal.

        :param publisher: ROS publisher for the specified servo.
        :param signal: Calculated control signal for the servo.
        :param axis: Axis of the servo being controlled ("horizontal" or
                        "vertical").
        """
        try:
            publisher.publish(Int32(data=signal))
            rospy.loginfo("Published %s signal: %d", axis, signal)
        except rospy.ROSException as e:
            rospy.logerr("Failed to publish %s signal: %s", axis, str(e))

    def close(self) -> None:
        """
        Clean up the CommunicationEspCam object.
        """
        self.horizontal_publisher.unregister()
        self.vertical_publisher.unregister()
        rospy.loginfo("CommunicationEspCam closed.")
