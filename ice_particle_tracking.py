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


def draw_tracks(img, track_data):
    """
    Function to draw tracking results to image
    :param img: OpenCV grayscale image
    :param track_data: ParticleTracks object
    """
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (0, 128, 255), (255, 0, 255)]
    color_index = 0
    for centroids in track_data.centroids:
        color = colors[color_index]
        for i in range(len(centroids[0])):
            cv2.circle(rgb_img, (int(centroids[0][i]), int(centroids[1][i])), 2, color, -1)
        color_index = (color_index + 1) % len(colors)
    cv2.imshow("Tracking Result", rgb_img)
    cv2.waitKey(0)


def ice_particle_tracking(input_directory: str, filter_by_length: bool = True, filter_by_velocity: bool = True,
                          filter_by_linearity: bool = True, min_length: int = 10, min_velocity: float = 5.0,
                          min_p: float = 0.99, model_spec=ModelPreset.constant_velocity_and_static_box_size_2d.value,
                          show_result: bool = True):
    """
    Function to performed motion-based multiple-object tracking of shed ice particles
    :param input_directory: path to input directory containing a series of undistorted PNG files
    :param filter_by_length: flag to filter tracking results by length
    :param filter_by_velocity: flag to filter tracking results by velocity
    :param filter_by_linearity: flag to filter tracking results by pearson correlation coefficient
    :param min_length: threshold length for filter
    :param min_velocity: threshold velocity for filter
    :param min_p: threshold correlation coefficient for filter
    :param model_spec: motpy tracking model
    :param show_result: flag for showing results of tracking script
    :return:
    """
    tracker = MultiObjectTracker(
        dt=0.1,
        model_spec=model_spec,
        matching_fn=IOUAndFeatureMatchingFunction(min_iou=0.1),
        active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 2},
        tracker_kwargs={'max_staleness': 2})
    detector = Detector()
    track_data = ParticleTracks()

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            img, current_frame = read_img_and_frame(os.path.join(input_directory, filename))
            detections = detector.detect(img)
            active_tracks = tracker.step(detections)
            active_tracks = track_data.delete_stalled_tracks(active_tracks, current_frame, min_velocity)
            track_data.update(active_tracks, img, current_frame)

    cv2.destroyAllWindows()
    filter_criteria = track_data.FilterCriteria(filter_by_length, filter_by_velocity, filter_by_linearity,
                                                min_length, min_velocity, min_p)
    track_data = track_data.filter_tracks(filter_criteria)
    if show_result:
        img, _ = read_img_and_frame(os.path.join(input_directory, filename))
        draw_tracks(img, track_data)


if __name__ == "__main__":
    ice_particle_tracking("C:/Users/JohannesBurger/AIIS/3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full"
                         "-Scale_Propeller/02_Data/Calib2/10/ChronosRGB/SE_01/PNG_PP",
                          filter_by_length=True, filter_by_velocity=True, filter_by_linearity=True,
                          min_length=10, min_velocity=5.0, min_p=0.99, show_result=True)
