from motpy.tracker import Detection
import cv2
import numpy as np


class Detector(object):
    """
    Detector class to detect objects in video frame
    Attributes:
        None
    """

    def __init__(self):
        """
        Initialize variables used by Detector class
        Args:
            None
        Return:
            None
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def detect(self, frame, min_blob_radius: float = 2.0, max_blob_radius: float = 15.0):
        """
        Detect objects in video frame using following pipeline
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
            - Find contours
            - Find centroids and draw box around minimum enclosing circle of contours
        Args:
            frame: single video frame
            min_blob_radius: minimum radius of detected blob in px
            max_blob_radius: maximum radius of detected blob in px
        Return:
            detections: list of Detection class objects containing bound boxes of detected elements
        """
        foreground_mask = self.background_subtractor.apply(frame)
        edges = cv2.Canny(foreground_mask, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if min_blob_radius < radius < max_blob_radius:
                radius = radius * 2
                box = np.array([x - radius, y - radius, x + radius, y + radius])
                detections.append(Detection(box))
        return detections
