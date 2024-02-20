import os
import cv2
import numpy as np
from motpy import ModelPreset, MultiObjectTracker, IOUAndFeatureMatchingFunction
from motpy.testing_viz import draw_track
from Detector import Detector
from ParticleTracks import ParticleTracks


def mb_particle_tracking(input_directory: str, filter_by_length: bool = True, filter_by_velocity: bool = True,
                         filter_by_linearity: bool = True, min_length: int = 10, min_velocity: float = 5.0,
                         min_p: float = 0.99, model_spec=ModelPreset.constant_velocity_and_static_box_size_2d.value):
    dt = 0.1
    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=model_spec,
        matching_fn=IOUAndFeatureMatchingFunction(min_iou=0.1),
        active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 2},
        tracker_kwargs={'max_staleness': 2})
    detector = Detector()
    track_data = ParticleTracks()

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(input_directory, filename), cv2.IMREAD_GRAYSCALE)
            if "_" in filename:
                filename = filename.split("_", 1)[1]
            current_frame = int(filename[:-4])
        detections = detector.detect(img)
        active_tracks = tracker.step(detections)

        for track in active_tracks:
            draw_track(img, track)
            if track.id in track_data.track_id:
                idx = track_data.track_id.index(track.id)
                xy = np.array([np.mean([track.box[0], track.box[2]]), np.mean([track.box[1], track.box[3]])])
                ds = np.linalg.norm(track_data.centroids[idx][:, -1] - xy)
                df = current_frame - track_data.frames[idx][-1]
                if ds / df < min_velocity:
                    del track
                else:
                    track_data.frames[idx] = np.append(track_data.frames[idx], current_frame)
                    track_data.centroids[idx] = np.array(
                        [np.append(track_data.centroids[idx][0, :], xy[0]),
                         np.append(track_data.centroids[idx][1, :], xy[1])])
            else:
                track_data.track_id.append(track.id)
                track_data.frames.append(np.array([current_frame]))
                track_data.centroids.append(
                    np.array([[np.mean([track.box[0], track.box[2]])], [np.mean([track.box[1], track.box[3]])]]))

        cv2.imshow('preview', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    filter_criteria = track_data.FilterCriteria(filter_by_length, filter_by_velocity, filter_by_linearity,
                                                min_length, min_velocity, min_p)
    track_data = track_data.filter_tracks(filter_criteria)
    color = 255
    filename = os.listdir(input_directory)[0]
    img = cv2.imread(os.path.join(input_directory, filename))
    for centroids in track_data.centroids:
        x_points = np.array(centroids[0]).astype('uint32')
        y_points = np.array(centroids[1]).astype('uint32')
        img[y_points.T, x_points.T] = color
    cv2.imshow('Result', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    mb_particle_tracking("C:/Users/JohannesBurger/AIIS/3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full"
                         "-Scale_Propeller/02_Data/Calib0/3/ChronosMono/SE_01/PNG_PP",
                         filter_by_length=True, filter_by_velocity=True, filter_by_linearity=True,
                         min_length=15, min_velocity=10.0, min_p=0.9975)
