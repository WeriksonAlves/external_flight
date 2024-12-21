from myLibs.rospy_uav.modules.Bebop2 import Bebop2
from myLibs.rospy_uav.modules.utils.DrawGraphics import DrawGraphics
from typing import List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt


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
