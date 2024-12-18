from ..auxiliary.MyTimer import TimingDecorator, TimeTracker
from ..interfaces.ClassifierInterface import ClassifierInterface
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Tuple
import numpy as np
import rospy


class KNN(ClassifierInterface):
    """
    KNN Classifier wrapper around sklearn's KNeighborsClassifier, with added
    timing and custom logic for training, predicting, and validating.
    """

    def __init__(self, model: KNeighborsClassifier) -> None:
        """
        Initialize the KNN classifier with a given KNeighborsClassifier
        instance.

        :param model: Instance of sklearn.neighbors.KNeighborsClassifier.
        :raises ValueError: If the model is not an instance of
                            KNeighborsClassifier.
        """
        if not isinstance(model, KNeighborsClassifier):
            raise ValueError(
                f"Expected KNeighborsClassifier, got {type(model).__name__}"
            )
        self.model = model

    @TimingDecorator.timing(use_cv2=True, log_output=False)
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the KNN model using the provided training data.

        :param x_train: Feature data for training (numpy array).
        :param y_train: Corresponding target labels for training (numpy array).
        :raises ValueError: If either x_train or y_train are not numpy arrays.
        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train,
                                                                 np.ndarray):
            raise ValueError("Both x_train and y_train must be numpy arrays.")

        rospy.loginfo(f"Training KNN model with {x_train.shape[0]} samples.")
        self.model.fit(x_train, y_train)

    def predict(self, reduced_data: np.ndarray, prob_min: float = 0.6) -> str:
        """
        Predict the class label for a given input sample based on probability.

        :param reduced_data: Input data sample (numpy array).
        :param prob_min: Minimum probability threshold for classification
                            (default: 0.6).
        :return: Predicted class label as a string, or 'Z' if below the
                    threshold.
        :raises ValueError: If reduced_data is not a numpy array.
        """
        if not isinstance(reduced_data, np.ndarray):
            raise ValueError("reduced_data must be a numpy array")

        reduced_data = reduced_data.flatten().reshape(1, -1)
        probabilities = self.model.predict_proba(reduced_data)

        # Check if the maximum probability exceeds the threshold
        predicted_class = self.model.predict(reduced_data)[0]
        if np.max(probabilities) > prob_min:
            return predicted_class
        return 'Z'

    @TimingDecorator.timing(use_cv2=True, log_output=False)
    def validate(self, X_val: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Validate the KNN model on a validation dataset.

        :param X_val: Validation data (numpy array).
        :return: Tuple of predicted class labels and their corresponding
                    classification times.
        :raises ValueError: If X_val is not a numpy array.
        """
        if not isinstance(X_val, np.ndarray):
            raise ValueError("X_val must be a numpy array")

        rospy.loginfo(f"Validating KNN model on {X_val.shape[0]} samples.")

        predictions = []
        classification_times = []

        # Vectorized approach for prediction and time measurement
        for sample in X_val:
            start_time = TimeTracker.get_current_time()
            predictions.append(self.predict(sample))
            elapsed_time = TimeTracker.calculate_elapsed_time(start_time)
            classification_times.append(elapsed_time)

        rospy.loginfo("Validation completed.")
        return predictions, classification_times
