#!/usr/bin/env python

# Internal library imports
from modules import (
    GestureRecognitionSystem,
    StandardCameras,
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN,
    FactoryMode
)

# External library imports
from sklearn.neighbors import KNeighborsClassifier
from typing import Union, Optional
import mediapipe as mp
import os
import rospy


# Constants
DATABASE_FILE = "datasets/DataBase_(5-10)_99.json"
DATABASE_FILES = [
    'datasets/DataBase_(5-10)_G.json',
    'datasets/DataBase_(5-10)_H.json',
    'datasets/DataBase_(5-10)_L.json',
    'datasets/DataBase_(5-10)_M.json',
    'datasets/DataBase_(5-10)_T.json',
    'datasets/DataBase_(5-10)_1.json',
    'datasets/DataBase_(5-10)_2.json',
    'datasets/DataBase_(5-10)_3.json',
    'datasets/DataBase_(5-10)_4.json',
    'datasets/DataBase_(5-10)_5.json',
    'datasets/DataBase_(5-10)_6.json',
    'datasets/DataBase_(5-10)_7.json',
    'datasets/DataBase_(5-10)_8.json',
    'datasets/DataBase_(5-10)_9.json',
    'datasets/DataBase_(5-10)_10.json'
]
NAME_VAL = "Val99"


def initialize_modes(mode: int) -> object:
    """
    Initializes the operation mode for the gesture recognition system.

    :param mode: Operation mode identifier (1=dataset, 2=validation,
                    3=real-time).
    :return: An instance of the configured operation mode.
    """
    rospy.loginfo(f"Initializing operation mode {mode}...")

    database_empty = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}
    try:
        if mode == 1:
            return FactoryMode.create_mode(
                mode_type='dataset',
                database=database_empty,
                file_name_build=DATABASE_FILE
            )
        elif mode == 2:
            return FactoryMode.create_mode(
                mode_type='validate',
                files_name=DATABASE_FILES,
                database=database_empty,
                name_val=NAME_VAL
            )
        elif mode == 3:
            return FactoryMode.create_mode(
                mode_type='real_time',
                files_name=DATABASE_FILES,
                database=database_empty
            )
        else:
            raise ValueError("Invalid mode. Supported modes are 1, 2, or 3.")
    except Exception as e:
        rospy.logerr(f"Error initializing operation mode: {e}")
        raise


def create_gesture_recognition_system(
    camera: Union[int, str], operation_mode: object, sps: Optional[object]
) -> GestureRecognitionSystem:
    """
    Creates and initializes the Gesture Recognition System.

    :param camera: Camera configuration (e.g., camera index or stream URL).
    :param operation_mode: The configured operation mode.
    :param sps: Servo position system, if applicable.
    :return: An instance of the GestureRecognitionSystem.
    """
    rospy.loginfo("Creating Gesture Recognition System...")
    try:
        return GestureRecognitionSystem(
            base_dir=os.path.dirname(__file__),  # Get the current folder
            configs=StandardCameras(camera, fps=15),  # Camera configuration
            operation_mode=operation_mode,
            tracker_model=MyYolo('yolov8n-pose.pt'),
            hand_extractor_model=MyHandsMediaPipe(
                mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.75
                )
            ),
            body_extractor_model=MyPoseMediaPipe(
                mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.75
                )
            ),
            classifier_model=KNN(
                KNeighborsClassifier(
                    n_neighbors=getattr(operation_mode, 'k', 5),
                    algorithm='auto',
                    weights='uniform'
                )
            ) if hasattr(operation_mode, 'k') else None,
            sps=sps
        )
    except Exception as e:
        rospy.logerr(f"Error creating Gesture Recognition System: {e}")
        raise


def initialize_camera(camera: str) -> Union[int, str]:
    """
    Initializes the camera configuration based on the input.

    :param camera: The camera type ('realsense' or 'espcam').
    :return: Camera index (int) or stream URL (str).
    """
    rospy.loginfo(f"Initializing camera: {camera}...")
    if camera.lower() == 'realsense':
        return 4
    elif camera.lower() == 'espcam':
        return "http://192.168.209.199:81/stream"
    else:
        rospy.logwarn(
            f"Unknown camera type '{camera}', defaulting to 'realsense'."
        )
        return 4


def main() -> None:
    """
    Main function to initialize and run the Gesture Recognition System.
    """
    rospy.init_node('Gesture_Recognition', anonymous=True)

    try:
        # Set the operation mode
        operation_mode = initialize_modes(mode=3)

        # Initialize and run the system
        gesture_system = create_gesture_recognition_system(
            camera=initialize_camera('realsense'),
            operation_mode=operation_mode,
            sps=None
        )

        rospy.loginfo("\n\nStarting Gesture Recognition System...\n\n")
        gesture_system.run()

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
    finally:
        if 'gesture_system' in locals():
            gesture_system.terminate()
        rospy.loginfo("Gesture Recognition System stopped.")


if __name__ == "__main__":
    main()
