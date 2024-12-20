#!/usr/bin/env python

# %% Imports
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Optional, Union
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
)

# UAV Library Imports
from myLibs.rospy_uav.modules import Bebop2


# %% Gesture Recognition System
class GestureRecognition:
    """
    A class to initialize and configure the Gesture Recognition System (GRS).
    """

    def __init__(
        self, database_file: str, database_files: List[str], name_val: str
    ) -> None:
        """
        Constructor for InitializeGRS.

        :param database_file: Path to the primary database file.
        :param database_files: List of database files for validation or
                                training.
        :param name_val: Name or identifier for the validation dataset.
        """
        self.database_file = database_file
        self.database_files = database_files
        self.name_val = name_val
        self.system = None

    def operation_mode(self, mode: str) -> object:
        """
        Configure the operation mode for the Gesture Recognition System.

        :param mode: Operation mode identifier ('D'=dataset, 'V'=validation,
                        'R'=real-time).
        :return: Instance of the configured operation mode.
        :raises ValueError: If an invalid mode is provided.
        """
        rospy.loginfo(f"Configuring operation mode: {mode}")

        database_empty = {"F": [], "I": [], "L": [], "P": [], "T": []}
        mode_type_map = {
            'D': (
                "dataset",
                {
                    "database": database_empty,
                    "file_name_build": self.database_file},
            ),
            'V': (
                "validate",
                {
                    "files_name": self.database_files,
                    "database": database_empty,
                    "name_val": self.name_val,
                },
            ),
            'R': (
                "real_time",
                {
                    "files_name": self.database_files,
                    "database": database_empty},
            ),
        }

        if mode not in mode_type_map:
            rospy.logerr(
                f"Invalid mode '{mode}'. Supported modes are 'D', 'V', 'R'."
            )
            raise ValueError(
                "Invalid mode. Supported modes are 'D', 'V', or 'R'."
            )

        mode_type, kwargs = mode_type_map[mode]
        try:
            return FactoryMode.create_mode(mode_type=mode_type, **kwargs)
        except Exception as e:
            rospy.logerr(f"Error configuring operation mode: {e}")
            raise

    def create_gesture_recognition_system(
        self,
        base_dir: str,
        camera: Union[int, str, Bebop2],
        operation_mode: object,
        tracker_model: object,
        hand_extractor_model: object,
        body_extractor_model: object,
        classifier_model: Optional[KNN] = None,
    ) -> GRS:
        """
        Create an instance of the Gesture Recognition System (GRS).

        :param base_dir: Base directory for the GRS configuration.
        :param camera: Camera source (e.g., index, URL, or Bebop2 instance).
        :param operation_mode: Configured operation mode instance.
        :param tracker_model: Tracker model instance (e.g., YOLO-based).
        :param hand_extractor_model: Hand feature extraction model.
        :param body_extractor_model: Body feature extraction model.
        :param classifier_model: Optional gesture classifier model (e.g., KNN).
        :return: Configured GRS instance.
        """
        rospy.loginfo("Initializing Gesture Recognition System...")
        try:
            self.system = GRS(
                base_dir=base_dir,
                camera=CameraSetup(camera),
                configs=SettingParameters(fps=15),
                operation_mode=operation_mode,
                tracker_model=tracker_model,
                hand_extractor_model=hand_extractor_model,
                body_extractor_model=body_extractor_model,
                classifier_model=classifier_model,
            )
            return self.system
        except Exception as e:
            rospy.logerr(f"Failed to initialize GRS: {e}")
            raise


# %% Main Code
def main():
    rospy.init_node("External_Flight", anonymous=True)

    rospy.loginfo("Setting up operation mode...")
    database_file = "myLibs/grs/datasets/DataBase_(5-10)_99.json"
    database_files = [
        f"myLibs/grs/datasets/DataBase_(5-10)_{i}.json"
        for i in ["G", "H", "L", "M", "T", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]
    name_val = "Val99"

    # Initialize GRS setup
    initialize_grs = GestureRecognition(
        database_file, database_files, name_val
    )
    operation_mode = initialize_grs.operation_mode('R')

    rospy.loginfo("Initializing camera...")
    camera = Bebop2(drone_type='bebop2', ip_address='192.168.0.202')

    rospy.loginfo("Initializing Gesture Recognition System...")
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

    grs = initialize_grs.create_gesture_recognition_system(
        base_dir=base_dir,
        camera=camera,
        operation_mode=operation_mode,
        tracker_model=tracker_model,
        hand_extractor_model=hand_extractor_model,
        body_extractor_model=body_extractor_model,
        classifier_model=classifier_model,
    )

    rospy.loginfo("Initializing Bebop2 drone...")
    bebop = camera if isinstance(camera, Bebop2) else None

    rospy.loginfo("Starting Gesture Recognition System...")
    grs.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Gesture Recognition System.")
