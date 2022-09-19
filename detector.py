import abc
from abc import ABC, abstractmethod
from detected_object import DetectedObject
class Detector(ABC):
        
    @abstractmethod
    def __init__(self, model_threshold, model_path):
        """
        Args:
            model_threshold (float): Minimum score for instance prediction to be shown. Defaults to 0.5.
            model_path (str): Set model of object detector
        """
        pass
    
    @abstractmethod
    def detect_objects(self, image, classes = [0, 32]) -> [DetectedObject]:
        """
        Args:
            image (Mat): cv2.imread() or any array with image with compatible format like cv2
            classes (list): List of object class identifiers that will be detected
        Returns:
            list: List of DetectedObject objects
        """
        pass
