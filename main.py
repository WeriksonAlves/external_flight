#!/usr/bin/env python

# %% Imports
from modules import DroneManager, GestureRecognition
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import os
import rospy

# GRS Library Imports
from myLibs.grs.modules import (
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN,
)


def main():
    rospy.init_node("External_Flight", anonymous=True)

    rospy.loginfo("Setting up operation mode...")
    database_file = "myLibs/grs/datasets/DataBase_(5-10)_99.json"
    database_files = [
        f"myLibs/grs/datasets/DataBase_(5-10)_{i}.json"
        for i in ["G", "H", "L", "M", "T", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    name_val = "Val99"
    drone_type = "bebop2"
    ip_address = "192.168.0.202"

    drone_manager = DroneManager(drone_type, ip_address)
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

    gesture_recognition.initialize_grs(
        base_dir=base_dir,
        camera=drone_manager.uav,
        operation_mode=operation_mode,
        tracker_model=tracker_model,
        hand_extractor_model=hand_extractor_model,
        body_extractor_model=body_extractor_model,
        classifier_model=classifier_model,
    )

    rospy.loginfo("Starting Gesture Recognition System...")
    # Executed in the main thread
    gesture_recognition.system.run()

    # Execute in the auxiliar thread 1
    previous_command = None
    while not rospy.is_shutdown():
        command = gesture_recognition.get_latest_command()
        if command != previous_command:
            drone_manager.execute_command(command)
            previous_command = command


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognition System.")
