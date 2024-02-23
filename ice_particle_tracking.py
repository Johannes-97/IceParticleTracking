import os
import cv2
from motpy import ModelPreset, MultiObjectTracker, IOUAndFeatureMatchingFunction
from Detector import Detector
from ParticleTracks import ParticleTracks


def read_img_and_frame(filename):
    """
    Function to read in image and frame number from filename
    :param filename: filename of image to be read
    :return: OpenCV image object and (int) frame number
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    filename = filename.split("\\", 1)[1]
    if "_" in filename:
        filename = filename.split("_", 1)[1]
    return img, int(filename[:-4])


def draw_tracks(input_dir, track_data):
    """
    Function to draw tracking results to image
    :param input_dir: input directory containing the png files to be tracked
    :param track_data: ParticleTracks object
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            rgb_img = cv2.imread(os.path.join(input_dir,filename), cv2.IMREAD_COLOR)
            break
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
                          filter_criteria: ParticleTracks.FilterCriteria=ParticleTracks.FilterCriteria(),
                          show_result: bool = True):
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

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            img, current_frame = read_img_and_frame(os.path.join(input_directory, filename))
            detections = detector.detect(img)
            active_tracks = tracker.step(detections)
            particle_tracks.update(active_tracks, img, current_frame)
    cv2.destroyAllWindows()

    particle_tracks = particle_tracks.filter_tracks(filter_criteria)
    particle_tracks.save_to_json(input_directory)
    if show_result:
        draw_tracks(input_directory, particle_tracks)


if __name__ == "__main__":
    detector = Detector(compute_opening=False)
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
    filter_criteria = ParticleTracks.FilterCriteria(filter_by_length=True, filter_by_circle=True, filter_by_velocity=True, filter_by_linearity=True,
                          min_length=10, min_velocity=5.0, min_p=0.99)
    ice_particle_tracking("C:/Users/JohannesBurger/AIIS/3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full"
                         "-Scale_Propeller/02_Data/Calib2/10/ChronosMono/SE_01/PNG", detector, tracker, filter_criteria,
                          show_result=True)
