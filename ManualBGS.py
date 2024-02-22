import cv2
import numpy as np
from motpy import ModelPreset


class ManualBGS(object):
    """
    Detector class for manual object detection
    """

    def __init__(self, blob_radius: int = 5, model_spec=ModelPreset.constant_velocity_and_static_box_size_2d):
        """
        Initialise function for ManualBGS object
        :param blob_radius: radius of blobs to be drawn on circles
        """
        self.blob_radius = blob_radius
        self.model_spec = model_spec

    def apply(self, frame):
        h, w = frame.shape
        new_frame = frame.copy()
        ret_frame = np.zeros((h, w), dtype="uint8")

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(new_frame, (x, y), self.blob_radius, (255, 255, 255), -1)
                cv2.circle(ret_frame, (x, y), self.blob_radius, (255, 255, 255), -1)

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click_event)
        while True:
            cv2.imshow("Image", new_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                break
        cv2.destroyAllWindows()

        return ret_frame
