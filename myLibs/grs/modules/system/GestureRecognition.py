from .Settings import DatasetMode, ValidationMode, RealTimeMode
from ..auxiliary.MyDataHandler import MyDataHandler
from ..auxiliary.MyTimer import TimeTracker
from ..camera.MyCamera import CameraSetup
from ..camera.SettingParameters import SettingParameters
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from typing import Tuple, Union
import cv2
import numpy as np
import os
import rospy
import threading


class ModeManager:
    """
    Handles the initialization and management of different operation modes.
    """

    def __init__(
        self, configs: SettingParameters,
        operation_mode: Union[DatasetMode, ValidationMode, RealTimeMode]
    ) -> None:
        """
        Initializes the ModeManager.

        :param configs: Configuration settings for the system (e.g., camera
                        parameters).
        :param operation_mode: The operational mode (Dataset, Validation, or
                                RealTime).
        """
        self.configs = configs
        self.operation_mode = operation_mode

    def initialize_mode(self) -> bool:
        """
        Initializes the specified operational mode and associated variables.

        :return: True if initialization is successful, False otherwise.
        """
        try:
            mode_initializers = {
                'D': self._initialize_dataset_mode,
                'V': self._initialize_validation_mode,
                'RT': self._initialize_real_time_mode
            }
            mode_initializer = mode_initializers.get(self.operation_mode.task)

            if not mode_initializer:
                rospy.logerr(
                    "Invalid mode specified: %s", self.operation_mode.task
                )
                return False

            mode_initializer()  # Call the specific initializer
            rospy.loginfo(
                "ModeManager initialized with mode: %s",
                self.operation_mode.task
            )
            self._initialize_shared_variables()
            return True
        except Exception as e:
            rospy.logerr("Error initializing mode: %s", str(e))
            return False

    def _initialize_dataset_mode(self) -> None:
        """
        Initializes dataset collection mode-specific variables.
        """
        self.database = self.operation_mode.database
        self.file_name_build = self.operation_mode.file_name_build
        self.max_num_gest = self.operation_mode.max_num_gest
        rospy.loginfo("Dataset mode initialized.")

    def _initialize_validation_mode(self) -> None:
        """
        Initializes validation mode-specific variables.
        """
        self.database = self.operation_mode.database
        self.proportion = self.operation_mode.proportion
        self.files_name = self.operation_mode.files_name
        self.file_name_val = self.operation_mode.file_name_val
        rospy.loginfo("Validation mode initialized.")

    def _initialize_real_time_mode(self) -> None:
        """
        Initializes real-time gesture recognition mode-specific variables.
        """
        self.database = self.operation_mode.database
        self.proportion = self.operation_mode.proportion
        self.files_name = self.operation_mode.files_name
        rospy.loginfo("Real-time mode initialized.")

    def _initialize_shared_variables(self) -> None:
        """
        Initializes variables shared across all modes.
        """
        try:
            self.dist = self.configs.dist
            self.length = self.configs.length
            self.stage = 0
            self.num_gest = 0
            self.dist_point = 1.0
            self.prob_min = 0.6
            self.hand_results = None
            self.hand_history = None
            self.body_results = None
            self.body_history = None
            self.sample = None
            self.predictions = []
            self.time_classifier = []

            # Initialize storage variables
            self.initialize_storage_variables()

            rospy.loginfo("Shared system variables initialized.")
        except Exception as e:
            rospy.logerr("Error initializing shared variables: %s", str(e))

    def initialize_storage_variables(self) -> None:
        """
        Initializes storage variables for hand and pose data.
        """
        try:
            self.hand_history, self.body_history, self.sample = MyDataHandler.initialize_data(
                dist=self.dist,
                length=self.length
            )
            rospy.logdebug("Storage variables initialized.")
        except Exception as e:
            rospy.logerr("Error initializing storage variables: %s", str(e))


class DataManager:
    """
    Manages dataset initialization, classifier training, and validation
    processes.
    """

    def __init__(
        self, operation_mode: Union[DatasetMode, ValidationMode, RealTimeMode],
        base_dir: str, classifier: ClassifierInterface
    ) -> None:
        """
        Initializes the DataManager with the given operation mode, base
        directory, and classifier.

        :param operation_mode: The operational mode (Dataset, Validation, or
                                RealTime).
        :param base_dir: The base directory for datasets and result files.
        :param classifier: An instance of the classifier interface for
                            training and validation.
        """
        self.operation_mode = operation_mode
        self.base_dir = base_dir
        self.classifier = classifier

        # Shared variables
        self.target_names = None
        self.y_val = None
        self.predictions = []
        self.time_classifier = []

    def setup_mode(self) -> None:
        """
        Sets up functionalities specific to the operational mode.

        :raises ValueError: If the operation mode is invalid.
        """
        try:
            mode_actions = {
                'D': self._initialize_database,
                'RT': self._load_and_fit_classifier,
                'V': self._validate_classifier
            }
            action = mode_actions.get(self.operation_mode.task)

            if not action:
                rospy.logerr(
                    "Invalid mode specified: %s", self.operation_mode.task
                )
                raise ValueError(f"Invalid mode: {self.operation_mode.task}")

            action()
            rospy.loginfo(
                "DataManager initialized for mode: %s",
                self.operation_mode.task
            )

        except Exception as e:
            rospy.logerr("Error during mode setup: %s", str(e))
            raise

    def _initialize_database(self) -> None:
        """
        Initializes the gesture database by loading target names and
        validation labels.
        """
        try:
            self.target_names, self.y_val = MyDataHandler.initialize_database(
                self.operation_mode.database
            )
            rospy.loginfo(
                "Database initialized successfully for dataset: %s",
                self.operation_mode.database
            )
        except Exception as e:
            rospy.logerr("Failed to initialize database: %s", str(e))
            raise

    def _load_and_fit_classifier(self) -> None:
        """
        Loads the dataset and fits the classifier for real-time recognition.
        """
        try:
            x_train, y_train, _, _ = MyDataHandler.load_database(
                self.base_dir,
                self.operation_mode.files_name,
                self.operation_mode.proportion
            )
            self.classifier.fit(x_train, y_train)
            rospy.loginfo(
                "Classifier successfully trained for real-time recognition."
            )
        except Exception as e:
            rospy.logerr("Error loading and fitting classifier: %s", str(e))
            raise

    def _validate_classifier(self) -> None:
        """
        Validates the classifier using the provided dataset and saves the
        results.

        :raises Exception: If any error occurs during validation or result
                            saving.
        """
        try:
            # Load training and validation datasets
            x_train, y_train, x_val, self.y_val = MyDataHandler.load_database(
                self.base_dir,
                self.operation_mode.files_name,
                self.operation_mode.proportion
            )
            rospy.loginfo("Database loaded successfully for validation.")

            # Train the classifier
            self.classifier.fit(x_train, y_train)
            rospy.loginfo("Classifier trained for validation mode.")

            # Validate the classifier
            self.predictions, self.time_classifier = self.classifier.validate(
                x_val
            )
            rospy.loginfo(
                "Classifier validation completed. Predictions generated."
            )

            # Initialize target names and save results
            self.target_names, _ = MyDataHandler.initialize_database(
                self.operation_mode.database
            )
            result_path = os.path.join(
                self.base_dir, 'results', self.operation_mode.file_name_val
            )
            MyDataHandler.save_results(
                self.y_val.tolist(),
                self.predictions,
                self.time_classifier,
                self.target_names,
                result_path
            )
            rospy.loginfo("Validation results saved to: %s", result_path)

        except Exception as e:
            rospy.logerr("Error during classifier validation: %s", str(e))
            raise


class DataAcquisition:
    """
    Handles real-time image acquisition using a separate thread. Captures
    frames from a camera and provides thread-safe access to the latest frame.
    """

    def __init__(self, configs: CameraSetup) -> None:
        """
        Initializes the DataAcquisition class.

        :param configs: CameraSetup configuration object containing the
                        `cap` (camera capture object).
        """
        if not configs or not configs.cap:
            rospy.logerr("Invalid camera configuration. 'cap' cannot be None.")
            raise ValueError("Camera configuration is required.")

        self.cap = configs.cap

        # Shared variables
        self.frame_captured = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.image_thread = None

        rospy.loginfo("DataAcquisition initialized successfully.")

    def start_image_thread(self) -> bool:
        """
        Starts the image acquisition thread.

        :return: True if the thread starts successfully, False otherwise.
        """
        if self.image_thread and self.image_thread.is_alive():
            rospy.logwarn("Image acquisition thread is already running.")
            return False

        try:
            self.image_thread = threading.Thread(
                target=self._read_image_thread, daemon=True
            )
            self.image_thread.start()
            rospy.loginfo("Image acquisition thread started.")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to start image thread: {e}")
            return False

    def _read_image_thread(self) -> None:
        """
        Thread function for reading and capturing images.
        Continuously reads frames from the camera and updates the latest frame.
        """
        rospy.logdebug("Image acquisition thread is running.")
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if success:
                resized_frame = self._resize_frame(frame)
                with self.frame_lock:
                    self.frame_captured = resized_frame
            else:
                cv2.imshow("Main Camera", np.zeros((480, 640, 3), np.uint8))
                rospy.logwarn("Failed to capture frame from the camera.")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resizes the frame to a standard resolution (640x480) if necessary.

        :param frame: Input frame to resize.
        :return: Resized frame.
        """
        target_resolution = (480, 640)  # Height x Width
        if frame.shape[:2] != target_resolution:
            rospy.logdebug(
                "Resizing frame from %s to %s", frame.shape[:2],
                target_resolution
            )
            return cv2.resize(frame, (640, 480))
        return frame

    def read_image(self) -> Tuple[bool, np.ndarray]:
        """
        Provides thread-safe access to the most recent captured frame.

        :return: Tuple (success, frame). `success` is True if a frame is
                    available, and `frame` is the captured frame or None if no
                    frame is available.
        """
        with self.frame_lock:
            success = self.frame_captured is not None
            return success, self.frame_captured

    def stop_image_thread(self) -> None:
        """
        Stops the image acquisition thread and releases resources.
        """
        self.stop_event.set()
        if self.image_thread and self.image_thread.is_alive():
            self.image_thread.join()
        rospy.loginfo("Image acquisition thread stopped.")

        # Release the camera resource
        if self.cap:
            self.cap.release()
            rospy.loginfo("Camera resource released.")


class TrackerProcessor:
    """
    Processes operator detection and tracking using a tracker model.
    """

    def __init__(
        self, tracker_model: TrackerInterface, show_logs: bool = False
    ) -> None:
        """
        Initializes the TrackerProcessor with the given tracker model.

        :param tracker_model: An implementation of the TrackerInterface to
                                handle person detection and tracking.
        """
        if not tracker_model:
            rospy.logerr("Tracker model cannot be None.")
            raise ValueError("Tracker model must be provided.")
        self.tracker = tracker_model
        self.show_logs = show_logs
        rospy.loginfo(
            "TrackerProcessor initialized with the given tracker model."
            )

    def process_tracking(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        """
        Processes a frame to detect, identify, and crop the operator.

        :param frame: Input video frame as a NumPy array.
        :return: Cropped image of the operator if successful, otherwise None.
        """
        if frame is None:
            rospy.logwarn("Received an empty frame. Skipping processing.")
            return None

        try:
            results_people, annotated_frame = self.tracker.detect_people(frame)
            rospy.logdebug("People detected in the frame: %s", results_people)

            self.bounding_box, track_id = self.tracker.identify_operator(
                results_people
            )
            if self.show_logs:
                rospy.loginfo(
                    "Operator identified with Track ID: %s", track_id
                )

            success, cropped_image = self.tracker.crop_operator(
                self.bounding_box, track_id, annotated_frame, frame
            )

            if success:
                if self.show_logs:
                    rospy.loginfo(
                        "Successfully cropped the operator from the frame."
                    )
                return cropped_image
            else:
                rospy.logwarn("Failed to crop the operator. Returning None.")
                return None

        except Exception as e:
            rospy.logerr(
                "Error during operator detection and tracking: %s", str(e)
            )
            return None


class ExtractionProcessor:
    """
    Handles feature extraction for hand and body tracking, manages gesture
    stages, and provides robust error handling.
    """

    def __init__(
        self, sample: dict, hand_extractor_model: ExtractorInterface,
        body_extractor_model: ExtractorInterface, mode_manager: ModeManager,
        show_logs: bool = False
    ) -> None:
        """
        Initializes the ExtractionProcessor with feature extraction models.

        :param hand_extractor_model: Model for extracting hand features.
        :param body_extractor_model: Model for extracting body features.
        """
        self.sample = sample  # Gesture configuration
        self.hand_extractor = hand_extractor_model
        self.body_extractor = body_extractor_model
        self.mode_manager = mode_manager
        self.show_logs = show_logs

        # Shared variables
        self.time_gesture = None
        rospy.loginfo("ExtractionProcessor initialized successfully.")

    def process_extraction(
        self, cropped_image: np.ndarray, gesture_duration: float = 4.0
    ) -> bool:
        """
        Processes feature extraction based on the current stage. Moves through
        gesture stages and handles transitions.

        :param cropped_image: Cropped input image for feature extraction.
        :return: True if extraction succeeds, False otherwise.
        """
        try:
            if self.mode_manager.stage == 0:
                if not self._process_hand_tracking(cropped_image):
                    return False
            elif self.mode_manager.stage == 1:
                if not self._process_body_tracking(cropped_image):
                    return False

            # Transition to the reduction stage if conditions are met
            if (self.mode_manager.stage == 1) and (
                TimeTracker.calculate_elapsed_time(
                    self.time_gesture
                ) > gesture_duration
            ):
                self._transition_to_reduction_stage()
            return True
        except Exception as e:
            rospy.logerr(f"Error during feature extraction: {e}")
            self._repeat_last_history_entry()
            return False

    def _transition_to_reduction_stage(self) -> None:
        """
        Transitions the processor to the reduction stage after sufficient time.
        """
        self.mode_manager.stage = 2
        self.mode_manager.sample[
            'time_gest'
        ] = TimeTracker.calculate_elapsed_time(
            self.time_gesture
        )
        self.mode_manager.sample[
            'time_classifier'
        ] = TimeTracker.get_current_time()
        if self.show_logs:
            rospy.loginfo("Transitioned to reduction stage.")

    def _repeat_last_history_entry(self) -> None:
        """
        Repeats the last valid entries in hand and body history to avoid
        crashes.
        """
        rospy.logwarn(
            "Repeating last valid entries in history due to extraction "
            "failure."
        )
        if self.mode_manager.hand_history.size > 0:
            self.mode_manager.hand_history = np.vstack((
                self.mode_manager.hand_history,
                self.mode_manager.hand_history[-1]
            ))
        if self.mode_manager.body_history.size > 0:
            self.mode_manager.body_history = np.vstack((
                self.mode_manager.body_history,
                self.mode_manager.body_history[-1]
            ))

    def _process_hand_tracking(self, cropped_image: np.ndarray) -> bool:
        """
        Processes hand feature extraction and updates the gesture stage if
        triggered.

        :param cropped_image: Cropped image containing the hand.
        """
        try:
            hand_results = self.hand_extractor.find_features(cropped_image)
            if hand_results.multi_hand_landmarks is None:
                rospy.logwarn("No hands detected in the frame.")
                return False

            annotated_frame = self.hand_extractor.draw_features(
                cropped_image, hand_results
            )
            self._annotate_image(annotated_frame)

            hand_ref = self.hand_extractor.calculate_reference_pose(
                hand_results,
                self.mode_manager.sample['joints_trigger_reference'],
                self.mode_manager.sample['joints_trigger']
            )
            hand_pose = self.hand_extractor.calculate_pose(
                hand_results,
                self.mode_manager.sample['joints_trigger']
            )
            hand_center = np.array([hand_pose.flatten() - hand_ref])
            self.mode_manager.hand_history = np.vstack((
                self.mode_manager.hand_history, hand_center
            ))

            self._check_gesture_trigger()
            return True
        except Exception as e:
            rospy.logerr(f"Error during hand tracking: {e}")
            self._repeat_last_history_entry()
            return False

    def _check_gesture_trigger(self) -> None:
        """
        Checks if the gesture trigger is activated based on hand history.
        Updates the stage and resets timers if triggered.
        """
        (
            trigger,
            self.mode_manager.hand_history,
            self.mode_manager.dist_point
        ) = self._is_trigger_enabled(
            self.mode_manager.hand_history,
            self.mode_manager.sample['par_trigger_length'],
            self.mode_manager.sample['par_trigger_dist']
        )
        if trigger:
            self.mode_manager.stage = 1
            self.mode_manager.dist_point = 1
            self.time_gesture = TimeTracker.get_current_time()
            if self.show_logs:
                rospy.loginfo(
                    "Gesture trigger activated. Switching to phase 1."
                )

    def _is_trigger_enabled(
        self, history: np.ndarray, length: int, dist: float
    ) -> Tuple[bool, np.ndarray, float]:
        """
        Determines if a gesture trigger is enabled based on history length and
        distance.

        :param history: Historical data of hand positions.
        :param length: Minimum length of history required for trigger
                        detection.
        :param dist: Maximum allowed distance to activate the trigger.
        :return: Tuple (trigger_status, updated_history, calculated_distance).
        """
        if len(history) < length:
            return False, history, 1

        recent_history = history[-length:]
        mean_coords = np.mean(recent_history, axis=0).reshape(-1, 2)
        std_dev = np.std(mean_coords, axis=0)
        dist_point = np.sqrt(std_dev[0] ** 2 + std_dev[1] ** 2)

        rospy.logdebug(f"Gesture trigger distance calculated: {dist_point}")
        return dist_point < dist, recent_history, dist_point

    def _process_body_tracking(self, cropped_image: np.ndarray) -> bool:
        """
        Processes body feature extraction and updates the body history.

        :param cropped_image: Cropped image containing the body.
        """
        try:
            body_results = self.body_extractor.find_features(cropped_image)
            if body_results.pose_landmarks is None:
                rospy.logwarn("No body detected in the frame.")
                return False

            annotated_frame = self.body_extractor.draw_features(
                cropped_image, body_results
            )
            self._annotate_image(annotated_frame)

            body_ref = self.body_extractor.calculate_reference_pose(
                body_results,
                self.mode_manager.sample['joints_tracked_reference'],
                self.mode_manager.sample['joints_tracked'],
                dimensions=3
            )
            body_pose = self.body_extractor.calculate_pose(
                body_results, self.mode_manager.sample['joints_tracked'])
            body_center = np.array([body_pose.flatten() - body_ref])
            self.mode_manager.body_history = np.vstack((
                self.mode_manager.body_history, body_center
            ))
            return True
        except Exception as e:
            rospy.logerr(f"Error during body tracking: {e}")
            self._repeat_last_history_entry()
            return False

    def _annotate_image(self, frame: np.ndarray) -> None:
        """
        Annotates the frame with stage and distance information.

        :param frame: Frame to annotate.
        """
        annotation = (
            f"Stage: {self.mode_manager.stage} | "
            f"Dist: {1000*self.mode_manager.dist_point:.1f}"
        )
        if getattr(self, "mode", None) == "D":
            annotation += (
                f" | Gesture {self.num_gest + 1}: {self.y_val[self.num_gest]}"
            )
        cv2.putText(
            frame, annotation, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imshow("Main Camera", frame)
