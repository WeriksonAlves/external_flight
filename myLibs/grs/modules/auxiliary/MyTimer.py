from typing import Callable, Any
import cv2
import time
import functools
import rospy


class TimeTracker:
    """
    Strategy-based utility class for measuring and tracking the execution time
    of functions, supporting multiple time measurement methods.
    """

    @staticmethod
    def get_current_time(use_cv2: bool = True) -> float:
        """
        Capture the current time based on the selected method.

        :param use_cv2: Whether to use OpenCV's high-resolution clock (default:
                        True).
        :return: The current time in seconds.
        """
        return cv2.getTickCount() if use_cv2 else time.perf_counter()

    @staticmethod
    def calculate_elapsed_time(
        start_time: float, use_cv2: bool = True
    ) -> float:
        """
        Calculate the elapsed time since the start time.

        :param start_time: The recorded start time.
        :param use_cv2: Whether to use OpenCV's high-resolution clock (default:
                        True).
        :return: The elapsed time in seconds.
        """
        if use_cv2:
            return (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        return time.perf_counter() - start_time


class TimingDecorator:
    """
    A decorator-based utility for measuring and logging the execution time
    of functions.
    """

    @staticmethod
    def timing(use_cv2: bool = True, log_output: bool = True) -> Callable:
        """
        Decorator to measure and log the execution time of a function.

        :param use_cv2: Whether to use OpenCV's high-resolution clock for
                        timing (default: True).
        :param log_output: Flag to enable logging with rospy (default: True).
        :return: A wrapped function with timing measurements.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = TimeTracker.get_current_time(use_cv2)
                result = func(*args, **kwargs)
                elapsed = TimeTracker.calculate_elapsed_time(
                    start_time, use_cv2
                )

                fps = 1 / elapsed if elapsed > 0 else float('inf')
                message = (f"{func.__name__} executed in {elapsed:.5f} "
                           f"-> {fps:.0f} FPS")

                if log_output:
                    rospy.loginfo(message)
                else:
                    print(message)

                return result
            return wrapper
        return decorator
