import os
import cv2
from motpy import ModelPreset, MultiObjectTracker, IOUAndFeatureMatchingFunction
from Detector import Detector
from ParticleTracks import ParticleTracks
import glob
import numpy as np


def read_img_and_frame(image_path):
    """
    Function to read in image and frame number from image_path
    :param image_path: image_path of image to be read
    :return: OpenCV image object and (int) frame number
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, filename = os.path.split(image_path)
    if "_" in filename:
        filename = filename.split("_", 1)[1]
    return img, int(filename[:-4])


def draw_tracks(image_path, track_data):
    """
    Function to draw tracking results to image
    :param image_path: path to background image
    :param track_data: ParticleTracks object
    """
    rgb_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (0, 128, 255), (255, 0, 255)]
    color_index = 0
    for centroids in track_data.centroids:
        color = colors[color_index]
        for i in range(len(centroids[0])):
            cv2.circle(rgb_img, (int(centroids[0][i]), int(centroids[1][i])), 2, color, -1)
        color_index = (color_index + 1) % len(colors)

    cv2.namedWindow("Tracking Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking Result", 720, 540)
    cv2.imshow("Tracking Result", rgb_img, )
    cv2.waitKey(0)


def ice_particle_tracking(input_directory: str, detector: Detector=Detector(),
                          tracker: MultiObjectTracker = MultiObjectTracker(
                                                        dt=0.1,
                                                        model_spec=ModelPreset.constant_velocity_and_static_box_size_2d.value,
                                                        matching_fn=IOUAndFeatureMatchingFunction(min_iou=0.01),
                                                        active_tracks_kwargs={'min_steps_alive': 0, 'max_staleness': 1},
                                                        tracker_kwargs={'max_staleness': 2}),
                          filter_criteria: ParticleTracks.FilterCriteria=ParticleTracks.FilterCriteria(), show_result: bool = True):
    """
    Function to performed motion-based multiple-object tracking of shed ice particles
    :param input_directory: directory containing png files for tracking to be performed
    :param detector: Detector class object of particle detector
    :param tracker: MultiObjectTracker class object of motpy
    :param filter_criteria: ParticleTracks.FilterCriteria class object containing filter criteria
    :param show_result: flag for showing tracking results
    :return:
    """
    particle_tracks = ParticleTracks()

    filenames = glob.glob(f"{input_directory}//*.png")
    for filename in filenames:
        img, current_frame = read_img_and_frame(filename)
        detections = detector.detect(img)
        active_tracks = tracker.step(detections)
        particle_tracks.update(active_tracks, img, current_frame)
    cv2.destroyAllWindows()

    particle_tracks = particle_tracks.filter_tracks(filter_criteria)
    particle_tracks.save_to_json(input_directory)
    if show_result:
        draw_tracks(filenames[0], particle_tracks)


if __name__ == "__main__":
    brightness_mask = np.load(r"C:\Users\JohannesBurger\AIIS\3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full-Scale_Propeller\02_Data\Calib2\Calibration\ChronosRGB\mask.npy")
    detector = Detector(compute_opening=False, brightness_mask=brightness_mask)
    model_spec = {
        'order_pos': 1, 'dim_pos': 2,  # position is a center in 2D space; under constant velocity model
        'order_size': 0, 'dim_size': 2,  # bounding box is 2 dimensional; under constant velocity model
        'q_var_pos': 1.,  # process noise
        'r_var_pos': 0.01  # measurement noise
    }
    tracker = MultiObjectTracker(
        dt=0.1,
        model_spec=model_spec,
        matching_fn=IOUAndFeatureMatchingFunction(min_iou=0.01),
        active_tracks_kwargs={'min_steps_alive': 0, 'max_staleness': 1},
        tracker_kwargs={'max_staleness': 2})
    filter_criteria = ParticleTracks.FilterCriteria(filter_by_length=True, filter_by_circle=False, filter_by_velocity=True, filter_by_linearity=True,
                          min_length=5, min_velocity=5.0, min_p=0.99)
    ice_particle_tracking(r"C:\Users\JohannesBurger\AIIS\3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full"
                         r"-Scale_Propeller\02_Data\Calib2\11\ChronosRGB\SE_01\PNG_PP", detector, tracker, filter_criteria,
                          show_result=True)
