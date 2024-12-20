from myLibs.rospy_uav.modules import Bebop2
import rospy


class DroneManager:
    """
    A class to manage the UAV (Unmanned Aerial Vehicle) and handle
    drone-related operations.
    """

    def __init__(self, drone_type: str, ip_address: str, ) -> None:
        """
        Initialize DroneManager instance.

        :param drone_type: Type of the drone (e.g., 'bebop2').
        :param ip_address: IP address of the drone.
        """
        self.uav = Bebop2(drone_type, ip_address)
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

        :param command: The command to be executed (e.g., 'T', 'L', 'P', 'F',
                        'I').
        """
        if self.uav.drone_type == "bebop2":
            command_map = {
                'T': self.uav.takeoff,
                'L': self.uav.land,
                'P': self.uav.take_snapshot,
                'F': self.uav.follow_me,
                'I': self.uav.rotate,
            }
        elif self.uav.drone_type == "gazebo":
            command_map = {
                'T': self.uav.takeoff,
                'L': self.uav.land,
                'P': self.uav.fly_direct,
                'F': self.uav.flip,
                'I': self.uav.take_snapshot,
            }
        if command in command_map:
            rospy.loginfo(f"Executing command: {command}")
            # command_map[command]()
        else:
            rospy.loginfo("Unknown or inconclusive gesture command received.")
