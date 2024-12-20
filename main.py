#!/usr/bin/env python

from modules import GestureRecognition, PersonTracker, execute_flight_pattern, execute_trajectory, DroneTrajectoryManager
from myLibs.grs.modules import (
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN,
    DroneManager
)
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import os
import rospy
from myLibs.rospy_uav.modules import Bebop2


def main():
    rospy.init_node("External_Flight", anonymous=True)

    rospy.loginfo("Setting up operation mode...")
    database_file = "myLibs/grs/datasets/DataBase_(5-10)_99.json"
    database_files = [
        f"myLibs/grs/datasets/DataBase_(5-10)_{i}.json"
        for i in ["G", "H", "L", "M", "T", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    name_val = "Val99"
    drone_type = "gazebo"
    ip_address = "192.168.0.202"

    # Initialize Gesture Recognition System
    gesture_recognition = GestureRecognition(
        database_file, database_files, name_val
    )
    operation_mode = gesture_recognition.configure_operation_mode('R')

    base_dir = os.path.dirname(__file__)
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

    uav = Bebop2(drone_type, ip_address)
    if drone_type == "bebop2":
        command_map = {
            'T': uav.takeoff,
            'L': uav.land,
            'P': uav.take_snapshot,
            'F': None,
            'I': None,
        }
    elif drone_type == "gazebo":
        command_map = {
            'T': uav.takeoff,
            'L': uav.land,
            'P': execute_trajectory(DroneTrajectoryManager(uav), "cube"),
            'F': execute_trajectory(DroneTrajectoryManager(uav), "ellipse"),
            'I': execute_trajectory(DroneTrajectoryManager(uav), "lemniscate"),
        }
    drone_manager = DroneManager(uav, command_map)

    gesture_recognition.initialize_grs(
        base_dir=base_dir,
        camera=4,  # drone_manager.uav,
        operation_mode=operation_mode,
        tracker_model=tracker_model,
        hand_extractor_model=hand_extractor_model,
        body_extractor_model=body_extractor_model,
        classifier_model=classifier_model,
        drone_manager=drone_manager
    )

    # Start Threads
    rospy.loginfo(
        "Starting threads for gesture recognition and drone control..."
    )
    gesture_recognition.system.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognition System.")
