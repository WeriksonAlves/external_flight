from .auxiliary.MyDataHandler import MyDataHandler
from .auxiliary.MyTimer import TimingDecorator, TimeTracker
from .camera.MyCamera import CameraSetup
from .camera.SettingParameters import SettingParameters
from .interfaces.ClassifierInterface import ClassifierInterface
from .interfaces.ExtractorInterface import ExtractorInterface
from .interfaces.TrackerInterface import TrackerInterface
from .system.GestureRecognition import DataAcquisition
from .system.GestureRecognition import DataManager
from .system.GestureRecognition import ExtractionProcessor
from .system.GestureRecognition import ModeManager
from .system.GestureRecognition import TrackerProcessor
from .system.ServoPosition import ServoPosition
from .system.Settings import DatasetMode
from .system.Settings import ValidationMode
from .system.Settings import RealTimeMode
from typing import Optional, Union
import cv2
import numpy as np
import os
import rospy


class GRS:
    """
    Main class for the gesture recognition system, integrating mode handling,
    data management, real-time image acquisition, tracking, and feature
    extraction.
    """

    def __init__(
        self,
        base_dir: str,
        camera: CameraSetup,
        configs: SettingParameters,
        operation_mode: Union[DatasetMode, ValidationMode, RealTimeMode],
        tracker_model: TrackerInterface,
        hand_extractor_model: ExtractorInterface,
        body_extractor_model: ExtractorInterface,
        classifier_model: Optional[ClassifierInterface] = None,
        sps: Optional[ServoPosition] = None,
    ) -> None:
        """
        Initializes the Gesture Recognition System.
        :param base_dir: Base directory for file operations.
        :param configs: Configuration object for camera settings.
        :param operation_mode: Selected mode of operation (Dataset, Validation,
                                Real-Time).
        :param tracker_model: Tracker model for operator tracking.
        :param hand_extractor_model: Model for hand gesture extraction.
        :param body_extractor_model: Model for body movement extraction.
        :param classifier_model: Classifier model for gesture recognition.
        :param sps: Optional servo position controller for camera adjustments.
        """
        self.base_dir = base_dir
        self.camera = camera
        self.configs = configs
        self.operation_mode = operation_mode
        self.tracker = tracker_model
        self.hand_extractor = hand_extractor_model
        self.body_extractor = body_extractor_model
        self.classifier = classifier_model
        self.sps = sps

        # Initialize mode manager and data manager
        self.mode_manager = ModeManager(configs, operation_mode)
        self.data_manager = DataManager(
            operation_mode, base_dir, classifier_model
        )
        self.data_acquisition = DataAcquisition(camera)

        # Control loop flag
        self.loop = operation_mode.task != 'V'

        # System initialization
        if not self._initialize_system():
            rospy.logerr("System initialization failed.")
            raise RuntimeError(
                "Failed to initialize Gesture Recognition System."
                )

        # Initialize data acquisition and processing components
        self.tracker_processor = TrackerProcessor(tracker_model)
        self.extraction_processor = ExtractionProcessor(
            self.mode_manager.sample, hand_extractor_model,
            body_extractor_model, self.mode_manager
        )

    @TimingDecorator.timing()
    def _initialize_system(self) -> bool:
        """
        Initializes the system by setting up modes, data acquisition, and
        related components.

        :return: True if initialization succeeds, False otherwise.
        """
        try:
            self.mode_manager.initialize_mode()
            self.data_manager.setup_mode()
            self.data_acquisition.start_image_thread()
            rospy.loginfo(
                "Gesture Recognition System initialized successfully."
            )
            return True
        except Exception as e:
            rospy.logerr(f"Error during initialization: {e}")
            return False

    def run(self) -> None:
        """
        Runs the main loop for real-time gesture recognition or dataset
        collection.
        """
        try:
            frame_time = TimeTracker.get_current_time()

            while self.loop:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rospy.loginfo("Exit signal received (q pressed).")
                    self.terminate()
                    break

                if self._check_dataset_completion():
                    break

                if TimeTracker.calculate_elapsed_time(frame_time) > (
                    1 / self.configs.fps
                ):
                    frame_time = TimeTracker.get_current_time()
                    self._process_frame()

        except KeyboardInterrupt:
            rospy.loginfo("System interrupted by user.")
        finally:
            self.terminate()

    def terminate(self) -> None:
        """
        Terminates the system, releasing resources and stopping threads.
        """
        if not self.loop:
            rospy.logdebug(
                "Terminate called, but the system is already stopping."
            )
            return

        rospy.loginfo("Terminating Gesture Recognition System...")
        self.loop = False
        self.data_acquisition.stop_image_thread()
        self.camera.cap.release()
        cv2.destroyAllWindows()
        if self.sps:
            self.sps.close()
        rospy.loginfo("System terminated successfully.")

    def _check_dataset_completion(self) -> bool:
        """
        Checks if the maximum number of gestures for the dataset has been
        collected.

        :return: True if dataset collection is complete, False otherwise.
        """
        if (self.operation_mode.task == 'D') and (
            self.mode_manager.num_gest >= self.mode_manager.max_num_gest
        ):
            rospy.loginfo("Maximum number of gestures reached.")
            self.terminate()
            return True
        return False

    def _process_frame(self) -> None:
        """
        Processes a single frame from the camera for gesture recognition.
        """
        success, frame = self.data_acquisition.read_image()
        if not success:
            rospy.logwarn("No frame available for processing.")
            return

        if self.mode_manager.stage in [0, 1]:
            self._handle_tracking_stage(frame)

        if self.mode_manager.stage == 2:  # Preso na etapa anterior
            self._process_reduction_stage()

        if self.mode_manager.stage == 3:
            self.update_database()

        if self.mode_manager.stage == 4:
            self.classify_gestures(self.mode_manager.prob_min)

    # @TimingDecorator.timing()
    def _handle_tracking_stage(self, frame: np.ndarray) -> None:
        """
        Handles tracking and feature extraction during stages 0 and 1.
        """
        cropped_image = self.tracker_processor.process_tracking(frame)
        if cropped_image is not None:
            if not self.extraction_processor.process_extraction(cropped_image):
                cv2.imshow("Main Camera", cropped_image)
        else:
            cv2.imshow("Main Camera", frame)

    def _process_reduction_stage(self) -> None:
        """
        Processes data reduction in stage 2.
        """
        self.mode_manager.body_history = self.mode_manager.body_history[1:]
        self.mode_manager.sample[
            'data_pose_track'
        ] = self.mode_manager.body_history
        self.mode_manager.sample['data_reduce_dim'] = np.dot(
            self.mode_manager.body_history.T, self.mode_manager.body_history
        )
        if self.operation_mode.task == 'D':
            self.mode_manager.stage = 3
        else:
            self.mode_manager.stage = 4

    def update_database(self) -> None:
        """
        Updates the database with the current gesture data and resets the
        sample data.
        """
        self.mode_manager.sample[
            'data_pose_track'
        ] = self.mode_manager.body_history
        self.mode_manager.sample['answer_predict'] = self.data_manager.y_val[
            self.mode_manager.num_gest
        ]
        self._append_to_database()
        self._save_database_to_file()
        self.mode_manager.initialize_storage_variables()
        self.mode_manager.num_gest += 1
        self.mode_manager.stage = 0

    def _append_to_database(self) -> None:
        """
        Appends the current gesture sample to the database.
        """
        gesture_class = str(
            self.data_manager.y_val[self.mode_manager.num_gest]
        )
        self.mode_manager.database[gesture_class].append(
            self.mode_manager.sample
        )

    def _save_database_to_file(self) -> None:
        """
        Saves the current database to a file.
        """
        file_path = os.path.join(
            self.base_dir, self.operation_mode.file_name_build
        )
        MyDataHandler.save_database(
            self.mode_manager.sample, self.mode_manager.database, file_path
        )

    def classify_gestures(self, prob_min: float) -> None:
        """
        Classifies gestures in real-time mode and resets the sample data for
        the next classification.

        :param prob_min: Minimum probability threshold for classification.
        """
        predicted_class = self.classifier.predict(
            self.mode_manager.sample['data_reduce_dim'], prob_min
        )
        self.mode_manager.predictions.append(predicted_class)

        classification_time = TimeTracker.calculate_elapsed_time(
            self.mode_manager.sample['time_classifier']
        )
        self.mode_manager.time_classifier.append(classification_time)

        rospy.loginfo(
            f"The gesture belongs to class {predicted_class} "
            f"and took {classification_time:.3f}ms to classify."
        )
        self.mode_manager.initialize_storage_variables()
        self.mode_manager.stage = 0
