from myLibs.rospy_uav.modules import Bebop2
import rospy


class DroneManager:
    """
    A class to manage the UAV (Unmanned Aerial Vehicle) and handle
    drone-related operations.
    """

    def __init__(
        self, uav: Bebop2, cammand_map: dict
    ) -> None:
        """
        Initialize DroneManager instance.

        :param uav: An instance of the Bebop2 drone.
        :param cammand_map: A dictionary mapping gesture commands to drone
                            actions.
        """
        self.uav = uav
        self.command_map = cammand_map
        self.initialize_drone()

    def initialize_drone(self) -> None:
        """
        Initialize the drone by checking connection, battery status, and
        starting the video stream.
        """
        if (self.uav.drone_type == 'bebop2') and (
            not self.uav.check_connection()
        ):
            rospy.logerr(
                "Failed to connect to the drone. Please check the connection."
            )
            raise ConnectionError("Drone connection failed.")
        rospy.loginfo("Drone connected successfully.")

        battery_level = self.uav.sensor_manager.sensor_data.get(
            "battery_level", "Unknown"
        )
        rospy.loginfo(f"Battery Level: {battery_level}%")

        rospy.loginfo("Starting video stream...")
        self.uav.camera_on()
        self.uav.smart_sleep(1)

    def execute_command(self, command: str) -> None:
        """
        Execute the given command on the drone.

        :param command: The command to be executed.
        """
        if command in self.command_map:
            rospy.loginfo(f"Executing command: {command}")
            self.command_map[command]()
        else:
            rospy.loginfo("Unknown or inconclusive gesture command received.")
