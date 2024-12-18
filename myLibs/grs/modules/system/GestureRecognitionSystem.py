from .ServoPosition import ServoPosition
from .Settings import DatasetMode, ValidationMode, RealTimeMode
from ..auxiliary.MyDataHandler import MyDataHandler
from ..auxiliary.MyTimer import TimingDecorator, TimeTracker
from ..camera.StandardCameras import StandardCameras
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import os
import rospy
import threading


class GestureRecognitionSystem:
    """
    Main class for the gesture recognition system, integrating mode handling,
    data management, real-time image acquisition, tracking, and feature
    extraction.
    """

    def __init__(
        self,
        base_dir: str,
        configs: StandardCameras,
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
        self.data_acquisition = DataAcquisition(configs)

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
        self.configs.cap.release()
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


class ModeManager:
    """
    Handles the initialization and management of different operation modes.
    """

    def __init__(
        self, configs: StandardCameras,
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

    def __init__(self, configs: StandardCameras) -> None:
        """
        Initializes the DataAcquisition class.

        :param configs: StandardCameras configuration object containing the
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

            bounding_box, track_id = self.tracker.identify_operator(
                results_people
            )
            if self.show_logs:
                rospy.loginfo(
                    "Operator identified with Track ID: %s", track_id
                )

            success, cropped_image = self.tracker.crop_operator(
                bounding_box, track_id, annotated_frame, frame
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
