from typing import Dict, List, Type, Union
import os
import numpy as np
import rospy


class SingletonMeta(type):
    """Metaclass implementing the Singleton design pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class FactoryMode:
    """
    Factory class to create instances of operation modes.
    """

    @staticmethod
    def create_mode(
        mode_type: str, **kwargs
    ) -> Union["DatasetMode", "ValidationMode", "RealTimeMode"]:
        """
        Factory method to create instances of different mode classes based on
        the mode type.

        :param mode_type: Type of mode ('dataset', 'validate', 'real_time').
        :param kwargs: Parameters for initializing the mode class.
        :return: An instance of the corresponding mode class.
        """
        modes: Dict[str, Type] = {
            "dataset": DatasetMode,
            "validate": ValidationMode,
            "real_time": RealTimeMode,
        }

        if mode_type not in modes:
            rospy.logerr(f"Invalid mode type: {mode_type}")
            raise ValueError(f"Invalid mode type: {mode_type}")

        rospy.loginfo(f"Creating mode instance for type: {mode_type}")
        return modes[mode_type](**kwargs)


class DatasetMode:
    """
    Mode for handling datasets in gesture recognition.

    :param database: Dictionary containing gesture data.
    :param file_name_build: File name for building the dataset.
    :param max_num_gest: Maximum number of gestures to process.
    :param dist: Distance parameter for gesture processing.
    :param length: Length parameter for gesture processing.
    """

    def __init__(
        self, database: Dict[str, List], file_name_build: str,
        max_num_gest: int = 50, dist: float = 0.025, length: int = 15
    ) -> None:
        rospy.loginfo("Initializing DatasetMode...")
        self.task = "D"
        self.database = database
        self.file_name_build = file_name_build
        self.max_num_gest = max_num_gest
        rospy.loginfo(
            f"DatasetMode initialized with max_num_gest = {max_num_gest}"
        )


class ValidationMode:
    """
    Mode for validating gesture recognition models.

    :param files_name: List of files for validation.
    :param database: Dictionary containing gesture data.
    :param name_val: Name of the validation output file.
    :param proportion: Proportion of data used for validation.
    :param n_class: Number of gesture classes.
    :param n_sample_class: Number of samples per gesture class.
    """

    def __init__(
        self, files_name: List[str], database: Dict[str, List], name_val: str,
        proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10
    ) -> None:
        rospy.loginfo("Initializing ValidationMode...")
        self.task = "V"
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = self._calculate_k(n_class, n_sample_class)
        self.file_name_val = self._generate_validation_file_name(
            n_class, n_sample_class, name_val
        )
        rospy.loginfo(f"Validation file generated: {self.file_name_val}")

    def _calculate_k(self, n_class: int, n_sample_class: int) -> int:
        """
        Calculate the value of k based on the dataset size and parameters.
        """
        k = int(np.round(np.sqrt(
            len(self.files_name) * self.proportion * n_class * n_sample_class
        )))
        rospy.loginfo(f"Calculated k for validation mode: {k}")
        return k

    def _generate_validation_file_name(
        self, n_class: int, n_sample_class: int, name_val: str
    ) -> str:
        """
        Generate a validation file name based on parameters.

        :param n_class: Number of gesture classes.
        :param n_sample_class: Number of samples per gesture class.
        :param name_val: Base name of the validation file.
        :return: Full validation file name.
        """
        s = int(
            len(
                self.files_name
            ) * (
                1 - self.proportion
            ) * n_class * n_sample_class
        )
        ma_p = int(10 * self.proportion)
        me_p = int(10 * (1 - self.proportion))
        file_name = f"C{n_class}_S{s}_p{ma_p}{me_p}_k{self.k}_{name_val}"
        file_path = os.path.join("results", file_name)
        rospy.loginfo(f"Generated validation file name: {file_path}")
        return file_name


class RealTimeMode:
    """
    Mode for real-time gesture recognition.

    :param files_name: List of files for processing.
    :param database: Dictionary containing gesture data.
    :param proportion: Proportion of data used for processing.
    :param n_class: Number of gesture classes.
    :param n_sample_class: Number of samples per gesture class.
    """

    def __init__(
        self, files_name: List[str], database: Dict[str, List],
        proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10
    ) -> None:
        rospy.loginfo("Initializing RealTimeMode...")
        self.task = "RT"
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = self._calculate_k(n_class, n_sample_class)
        rospy.loginfo(f"Real-time mode initialized with k = {self.k}")

    def _calculate_k(self, n_class: int, n_sample_class: int) -> int:
        """
        Calculate the value of k for real-time processing.

        :param n_class: Number of gesture classes.
        :param n_sample_class: Number of samples per gesture class.
        :return: Computed value of k.
        """
        k = int(np.round(np.sqrt(
            len(self.files_name) * self.proportion * n_class * n_sample_class
        )))
        rospy.loginfo(f"Calculated k for real-time mode: {k}")
        return k
