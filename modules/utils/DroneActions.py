from myLibs.rospy_uav.modules.Bebop2 import Bebop2
from myLibs.rospy_uav.modules.utils.DrawGraphics import DrawGraphics
from typing import List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


class TrajectoryGenerator:
    """
    Generates predefined trajectories for drone operations.
    """

    @staticmethod
    def generate_cube() -> List[Tuple[float, float, float, float, float]]:
        """
        Generates a cube-shaped trajectory: (x, y, z, yaw, power).
        """
        return [
            (0, 0, 1, 0, 0.5),  # Vertex 1
            (1, 0, 1, 0, 0.5),  # Vertex 2
            (1, 1, 1, 0, 0.5),  # Vertex 3
            (0, 1, 1, 0, 0.5),  # Vertex 4
            (0, 0, 1, 0, 0.5),  # Return to Vertex 1
            (0, 0, 2, 0, 0.5),  # Vertex 5
            (1, 0, 2, 0, 0.5),  # Vertex 6
            (1, 1, 2, 0, 0.5),  # Vertex 7
            (0, 1, 2, 0, 0.5),  # Vertex 8
            (0, 0, 2, 0, 0.25),  # Return to Vertex 5
            (0, 0, 1, 0, 0.25),  # Return to Vertex 1
        ]

    @staticmethod
    def generate_ellipse(a: float = 2, b: float = 1, z: float = 3,
                         points: int = 50, seq: int = 1) -> List[Tuple[
                             float, float, float, float, float]]:
        """
        Generates an elliptical trajectory.
        :param a: Semi-major axis length.
        :param b: Semi-minor axis length.
        :param z: Fixed altitude.
        :param points: Number of points to generate.
        :param seq: Sequence number for the ellipse.
        :return: List of trajectory points (x, y, z, yaw, power).
        """
        return [
            (a * np.cos(seq * t), b * np.sin(seq * t), z, 0, 0.25)
            for t in np.linspace(0, 2 * np.pi, points)
        ]

    @staticmethod
    def generate_lemniscate(a=2, b=1, z=3, points=50
                            ) -> List[Tuple[float, float, float, float, float]
                                      ]:
        """
        Generates a lemniscate (figure-eight) trajectory.
        :param a: Horizontal scaling factor.
        :param b: Vertical scaling factor.
        :param z: Fixed altitude.
        :param points: Number of points to generate.
        """
        return [
            (a * np.sin(t), b * np.sin(2 * t), z, 0, 0.25)
            for t in np.linspace(0, 2 * np.pi, points)
        ]


class DroneTrajectoryManager:
    """
    Manages drone trajectory execution.
    """

    def __init__(self, drone: Bebop2) -> None:
        """
        Initialize the DroneTrajectoryManager with a drone instance.
        :param drone: Instance of the Bebop2 drone.
        """
        self.drone = drone

    def execute_trajectory(self, trajectory: List[Tuple[
            float, float, float, float, float]]) -> List[Tuple[
                float, float, float]]:
        """
        Executes a given trajectory and records drone positions.
        :param trajectory: List of trajectory points (x, y, z, yaw, power).
        :return: List of recorded drone positions.
        """
        positions = []
        for x, y, z, yaw, power in trajectory:
            self.drone.move_relative(x, y, z, yaw, power)
            self.drone.smart_sleep(0.1)
            odometry = self.drone.sensor_manager.get_sensor_data().get(
                "odometry", {}
            )
            position = tuple(odometry.get("position", (0, 0, 0)))
            positions.append(position)
        return positions


class FlightCommand:
    """
    Represents a single flight command for the drone.
    """

    def __init__(self, linear_x: int, linear_y: int, linear_z: int,
                 angular_z: int, duration: int) -> None:
        """
        Initializes a flight command.

        :param linear_x: Forward/backward speed.
        :param linear_y: Left/right speed.
        :param linear_z: Up/down speed.
        :param angular_z: Rotational speed.
        :param duration: Duration to execute the command.
        """
        self.linear_x = linear_x
        self.linear_y = linear_y
        self.linear_z = linear_z
        self.angular_z = angular_z
        self.duration = duration


class FlightPattern:
    """
    Manages predefined flight patterns for the drone.
    """

    @staticmethod
    def get_indoor_flight_pattern() -> List[FlightCommand]:
        """
        Returns a list of flight commands for an indoor flight pattern.

        :return: List of FlightCommand objects.
        """
        return [
            FlightCommand(0, 0, 50, 0, 3),  # Hover up
            FlightCommand(25, 0, 0, 0, 2),   # Move forward
            FlightCommand(-25, 25, 0, 0, 2),   # Move right
            FlightCommand(0, -25, 0, 0, 2),  # Move back and left
            FlightCommand(0, 0, 0, 50, 2),  # Rotate
            FlightCommand(0, 0, 0, -50, 2),  # Rotate
            FlightCommand(0, 0, -50, 0, 3),  # Hover down
        ]


class DroneActionManager:
    """
    Manages drone operations, including executing flight patterns.
    """

    def __init__(self, drone: Bebop2) -> None:
        """
        Initializes the DroneActionManager with a Bebop2 instance.

        :param drone: An instance of the Bebop2 drone.
        """
        self.drone = drone

    def execute_flight_pattern(self, pattern: List[FlightCommand]) -> None:
        """
        Executes a sequence of flight commands.

        :param pattern: List of FlightCommand objects.
        """
        print("Executing flight pattern...")
        for command in pattern:
            self.drone.fly_direct(
                linear_x=command.linear_x,
                linear_y=command.linear_y,
                linear_z=command.linear_z,
                angular_z=command.angular_z,
                duration=command.duration
            )
            self.drone.smart_sleep(command.duration)


def execute_flight_pattern(bebop: Bebop2) -> None:
    """
    Execute a predefined flight pattern for the drone.

    :param bebop: The Bebop2 drone instance.
    """
    # Create the drone action manager
    action_manager = DroneActionManager(bebop)

    # Execute flight pattern
    flight_pattern = FlightPattern.get_indoor_flight_pattern()
    action_manager.execute_flight_pattern(flight_pattern)


def execute_trajectory(
    trajectory_manager: DroneTrajectoryManager, trajectory_type: str
) -> None:
    """
    Generate and execute the selected trajectory.

    :param trajectory_manager: Instance of DroneTrajectoryManager.
    :param trajectory_type: Type of trajectory to generate and execute.
                            (Options: 'cube', 'ellipse', 'lemniscate')
    """
    # Generate and execute the selected trajectory
    trajectory_map: dict[str, Callable[..., List[Tuple[
        float, float, float, float, float]]]] = {
        "cube": TrajectoryGenerator.generate_cube,
        "ellipse": TrajectoryGenerator.generate_ellipse,
        "lemniscate": TrajectoryGenerator.generate_lemniscate,
    }

    if trajectory_type in trajectory_map:
        trajectory_func = trajectory_map[trajectory_type]
        trajectory_points = trajectory_func()
        print(f"Executing {trajectory_type} trajectory...")
        recorded_positions = trajectory_manager.execute_trajectory(
            trajectory_points
        )
        DrawGraphics.plot_trajectory(recorded_positions)
    else:
        print(f"Invalid trajectory type: {trajectory_type}. Options: 'cube', "
              "'ellipse', 'lemniscate'.")


def photografy(bebop: Bebop2, topic: str) -> None:
    """
    Take a snapshot with the drone's camera.

    :param bebop: The Bebop2 drone instance.
    :param topic: ROS topic to subscribe to for the image data.
    """
    ret, frame = bebop.read_image(topic)

    if not ret:
        print("Erro ao capturar a imagem. Verifique a conexão com o drone.")
        return

    snapshot_dir = os.path.join(os.getcwd(), 'images', 'snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_counter = len(os.listdir(snapshot_dir))
    snapshot_filename = f"img_{snapshot_counter:04d}.png"
    snapshot_filepath = os.path.join(snapshot_dir, snapshot_filename)

    cv2.imwrite(snapshot_filepath, frame)
    print(f"\nSnapshot salvo em: {snapshot_filepath}\n")


def follow_me(
    bebop: Bebop2, topic: str, bounding_box: np.ndarray,
    Gi: Tuple[float, float] = (0.5, 0.5), Ge: Tuple[float, float] = (50, 50)
) -> None:
    """
    Adjust the camera orientation and yaw based on the operator's position in the frame.

    :param uav: The Bebop2 drone instance.
    :param frame: The captured frame.
    :param bounding_box: Bounding box for the detected operator.
    :param Gi: Internal gain for pitch and yaw adjustment.
    :param Ge: External gain for pitch and yaw adjustment.
    """
    ret, frame = bebop.read_image(topic)

    if not ret:
        print("Erro ao capturar a imagem. Verifique a conexão com o drone.")
        return

    dist_center_h, dist_center_v = _centralize_operator(
        frame, bounding_box)

    sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0] if abs(
        dist_center_v) > 0.25 else 0
    sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] if abs(
        dist_center_h) > 0.25 else 0
    print(f"Pitch ajustado: {sc_pitch}, Yaw ajustado: {sc_yaw}")

    # Comando de voo: gira o drone para centralizar o operador no eixo horizontal
    bebop.fly_direct(0., 0., 0., dist_center_h, 0.1)
    
    # Move a câmera para centralizar o operador
    bebop.command_manager.adjust_camera_orientation(tilt=sc_pitch, pan=0)


def _centralize_operator(
    frame: np.ndarray, bounding_box: Tuple[int, int, int, int],
    drone_pitch: float = 0.0, drone_yaw: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate the horizontal and vertical distance from the bounding box to the frame's center.

    :param frame: The captured frame.
    :param bounding_box: The bounding box of the operator as (x, y, width, height).
    :param drone_pitch: The pitch value for compensation.
    :param drone_yaw: The yaw value for compensation.
    :return: Tuple[float, float]: The horizontal and vertical distance to the frame's center.
    """
    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    # Coordenadas da bounding box
    box_x, box_y, _, _ = bounding_box

    # Calcula o centro da bounding box
    box_center_x = box_x # + box_width // 2
    box_center_y = box_y # + box_height // 2

    # Calcula a distância para o centro do frame (normalizada)
    dist_to_center_h = (
        box_center_x - frame_center[0]) / frame_center[0] - drone_yaw
    dist_to_center_v = (
        box_center_y - frame_center[1]) / frame_center[1] - drone_pitch

    return dist_to_center_h, dist_to_center_v
