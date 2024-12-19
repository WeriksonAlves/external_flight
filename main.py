#!/usr/bin/env python

from typing import Union, Optional
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
import os
import rospy

# GRS Library Imports
from myLibs.grs.modules import (
    GRS,
    CameraSetup,
    SettingParameters,
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN,
    FactoryMode,
    ServoPosition,
)

# ROSPY_UAV Library Imports
from myLibs.rospy_uav.modules import Bebop2

# Constants
DATABASE_FILE = "myLibs/grs/datasets/DataBase_(5-10)_99.json"
DATABASE_FILES = [
    f"myLibs/grs/datasets/DataBase_(5-10)_{i}.json"
    for i in ["G", "H", "L", "M", "T", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]
NAME_VAL = "Val99"


def grs_operation_mode(mode: int) -> object:
    """
    Initializes the operation mode for the Gesture Recognition System.

    :param mode: Operation mode identifier (1=dataset, 2=validation,
                    3=real-time).
    :return: Configured operation mode instance.
    """
    rospy.loginfo(f"Initializing operation mode: {mode}")
    database_empty = {"F": [], "I": [], "L": [], "P": [], "T": []}

    try:
        mode_type_map = {
            1: (
                "dataset",
                {
                    "database": database_empty,
                    "file_name_build": DATABASE_FILE
                }
            ),
            2: (
                "validate",
                {
                    "files_name": DATABASE_FILES,
                    "database": database_empty,
                    "name_val": NAME_VAL,
                },
            ),
            3: (
                "real_time",
                {
                    "files_name": DATABASE_FILES,
                    "database": database_empty
                },
            ),
        }

        if mode not in mode_type_map:
            raise ValueError("Invalid mode. Supported modes are 1, 2, or 3.")

        mode_type, kwargs = mode_type_map[mode]
        return FactoryMode.create_mode(mode_type=mode_type, **kwargs)
    except Exception as e:
        rospy.logerr(f"Error initializing operation mode: {e}")
        raise


def initialize_camera(camera: str) -> Union[int, str, Bebop2]:
    """
    Initializes the camera configuration based on the input type.

    :param camera: Camera type ('realsense', 'espcam', or 'bebop').
    :return: Camera index, stream URL, or Bebop2 instance.
    """
    rospy.loginfo(f"Initializing camera: {camera}")
    camera_map = {
        "realsense": 4,
        "espcam": "http://192.168.209.199:81/stream",
        "bebop": Bebop2(drone_type="bebop2", ip_address="192.168.0.202"),
    }
    return camera_map.get(camera.lower(), 4)


def create_gesture_recognition_system(
    camera: Union[int, str, Bebop2],
    operation_mode: object,
    sps: Optional[ServoPosition] = None,
    bebop: Optional[Bebop2] = None,
) -> GRS:
    """
    Creates and initializes the Gesture Recognition System (GRS).

    :param camera: Camera configuration (e.g., camera index, URL, or Bebop2
                    instance).
    :param operation_mode: The configured operation mode.
    :param sps: Servo position system, if applicable.
    :param bebop: Bebop2 drone instance, if applicable.
    :return: Configured GRS instance.
    """
    rospy.loginfo("Creating Gesture Recognition System...")
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

        return GRS(
            base_dir=os.path.dirname(__file__),
            camera=CameraSetup(camera),
            configs=SettingParameters(fps=15),
            operation_mode=operation_mode,
            tracker_model=tracker_model,
            hand_extractor_model=hand_extractor_model,
            body_extractor_model=body_extractor_model,
            classifier_model=classifier_model,
            sps=sps,
            bebop=bebop,
        )
    except Exception as e:
        rospy.logerr(f"Error creating Gesture Recognition System: {e}")
        raise


def main() -> None:
    """
    Main function to initialize and run the Gesture Recognition System.
    """
    rospy.init_node("External_Flight", anonymous=True)

    try:
        rospy.loginfo("Setting up operation mode...")
        operation_mode = grs_operation_mode(mode=3)

        rospy.loginfo("Initializing camera...")
        camera = initialize_camera(camera="bebop")

        rospy.loginfo("Initializing Bebop2 drone...")
        bebop = camera if isinstance(camera, Bebop2) else None

        rospy.loginfo("Initializing Gesture Recognition System...")
        gesture_system = create_gesture_recognition_system(
            camera=camera, operation_mode=operation_mode, bebop=bebop
        )

        rospy.loginfo("Starting Gesture Recognition System...")
        gesture_system.run()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        if "gesture_system" in locals():
            gesture_system.terminate()
        rospy.loginfo("Gesture Recognition System stopped.")


if __name__ == "__main__":
    main()
