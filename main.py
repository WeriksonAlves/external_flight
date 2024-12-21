#!/usr/bin/env python

import os
import rospy
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp

# Custom Library Imports
from modules import (
    GestureRecognition,
    execute_trajectory,
    DroneTrajectoryManager,
)
from myLibs.grs.modules import (
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN,
    DroneManager,
)
from myLibs.rospy_uav.modules import Bebop2


def initialize_drone(drone_type: str, ip_address: str) -> Bebop2:
    """
    Initializes the UAV based on the provided drone type and IP address.

    :param drone_type: Type of the drone (e.g., "bebop2" or "gazebo").
    :param ip_address: IP address of the drone.
    :return: Initialized UAV object.
    """
    rospy.loginfo(f"Initializing drone: {drone_type}")
    try:
        return Bebop2(drone_type=drone_type, ip_address=ip_address)
    except Exception as e:
        rospy.logerr(f"Error initializing drone: {e}")
        raise


def create_command_map(
    drone_type: str, uav: Bebop2, trajectory_manager: DroneTrajectoryManager
) -> dict:
    """
    Creates a command map for the drone based on its type.

    :param drone_type: Type of the drone (e.g., "bebop2" or "gazebo").
    :param uav: UAV instance for command execution.
    :param trajectory_manager: Trajectory manager for executing predefined
                                paths.
    :return: Command map dictionary.
    """
    rospy.loginfo(f"Creating command map for drone type: {drone_type}")
    try:
        common_commands = {
            'T': uav.takeoff,
            'L': uav.land,
        }

        gazebo_commands = {
            'P': lambda: execute_trajectory(trajectory_manager, "cube"),
            'F': lambda: execute_trajectory(trajectory_manager, "ellipse"),
            'I': lambda: execute_trajectory(trajectory_manager, "lemniscate"),
        }

        bebop_commands = {
            'P': uav.take_snapshot,
            'F': None,  # Add specific action for Bebop if needed
            'I': lambda: execute_trajectory(trajectory_manager, "ellipse"),
        }

        return {
            **common_commands, **(
                gazebo_commands if drone_type == "gazebo" else bebop_commands
            )
        }
    except Exception as e:
        rospy.logerr(f"Error creating command map: {e}")
        raise


def configure_models(operation_mode: object) -> tuple:
    """
    Configures the models required for gesture recognition.

    :param operation_mode: Operation mode object containing system
                            configuration.
    :return: Tuple of tracker model, hand extractor model, body extractor
                model, and classifier model.
    """
    rospy.loginfo("Configuring models for gesture recognition...")
    try:
        tracker_model = MyYolo("yolov8n-pose.pt")

        hand_extractor_model = MyHandsMediaPipe(
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
            )
        )

        body_extractor_model = MyPoseMediaPipe(
            mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
            )
        )

        classifier_model = (
            KNN(
                KNeighborsClassifier(
                    n_neighbors=getattr(operation_mode, "k", 5),
                    algorithm="auto",
                    weights="uniform",
                )
            )
            if hasattr(operation_mode, "k")
            else None
        )

        return tracker_model, hand_extractor_model, body_extractor_model, classifier_model
    except Exception as e:
        rospy.logerr(f"Error configuring models: {e}")
        raise


def main():
    """
    Main function to initialize and run the Gesture Recognition System with
    drone control and gesture-based commands.
    """
    rospy.init_node("External_Flight", anonymous=True)

    rospy.loginfo("Setting up Gesture Recognition System...")
    database_file = "myLibs/grs/datasets/DataBase_(5-10)_99.json"
    database_files = [
        f"myLibs/grs/datasets/DataBase_(5-10)_{i}.json"
        for i in ["G", "H", "L", "M", "T", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    name_val = "Val99"
    drone_type = "gazebo"
    ip_address = "192.168.0.202"

    # Initialize gesture recognition and operation mode
    gesture_recognition = GestureRecognition(
        database_file, database_files, name_val
    )
    operation_mode = gesture_recognition.configure_operation_mode('R')

    # Configure models
    tracker_model, hand_extractor_model, body_extractor_model, classifier_model = configure_models(operation_mode)

    # Initialize UAV and trajectory manager
    uav = initialize_drone(drone_type, ip_address)
    trajectory_manager = DroneTrajectoryManager(uav)

    # Create command map and drone manager
    command_map = create_command_map(drone_type, uav, trajectory_manager)
    drone_manager = DroneManager(uav, command_map)

    # Initialize Gesture Recognition System
    base_dir = os.path.dirname(__file__)
    gesture_recognition.initialize_grs(
        base_dir=base_dir,
        camera=4,  # Replace with drone_manager.uav if needed
        operation_mode=operation_mode,
        tracker_model=tracker_model,
        hand_extractor_model=hand_extractor_model,
        body_extractor_model=body_extractor_model,
        classifier_model=classifier_model,
        drone_manager=drone_manager,
    )

    rospy.loginfo("Starting Gesture Recognition System and UAV control...")
    try:
        gesture_recognition.system.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognition System.")


if __name__ == "__main__":
    main()
