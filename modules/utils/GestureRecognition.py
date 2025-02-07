from typing import Union, Optional, List
import rospy

# GRS Library Imports
from myLibs.grs.modules import (
    GRS,
    CameraSetup,
    SettingParameters,
    KNN,
    FactoryMode,
    DroneManager,
)

# UAV Library Imports
from myLibs.rospy_uav.modules import Bebop2


class GestureRecognition:
    """
    A class to initialize and configure the Gesture Recognition System (GRS).
    """

    def __init__(
        self, database_file: str, database_files: List[str], name_val: str
    ) -> None:
        """
        Initialize GestureRecognition instance.

        :param database_file: Path to the primary database file.
        :param database_files: List of database files for validation or
                                training.
        :param name_val: Name or identifier for the validation dataset.
        """
        self.database_file = database_file
        self.database_files = database_files
        self.name_val = name_val
        self.system = None

    def configure_operation_mode(self, mode: str) -> object:
        """
        Configure the operation mode for the Gesture Recognition System.

        :param mode: Operation mode identifier ('D'=dataset, 'V'=validation,
                        'R'=real-time).
        :return: Configured operation mode instance.
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

    def initialize_grs(
        self,
        base_dir: str,
        camera: Union[int, str, Bebop2],
        operation_mode: object,
        tracker_model: object,
        hand_extractor_model: object,
        body_extractor_model: object,
        classifier_model: Optional[KNN] = None,
        drone_manager: Optional[DroneManager] = None,
    ) -> GRS:
        """
        Initialize the Gesture Recognition System (GRS).

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
                configs=SettingParameters(fps=30),
                operation_mode=operation_mode,
                tracker_model=tracker_model,
                hand_extractor_model=hand_extractor_model,
                body_extractor_model=body_extractor_model,
                classifier_model=classifier_model,
                drone_manager=drone_manager,
            )
            return self.system
        except Exception as e:
            rospy.logerr(f"Failed to initialize GRS: {e}")
            raise
