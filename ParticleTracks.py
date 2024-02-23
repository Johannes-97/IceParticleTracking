import numpy as np
from scipy.stats import pearsonr
from motpy.testing_viz import draw_track
import cv2
import json


class ParticleTracks(object):
    """
    Class to store particle track data
    track_id: list of track ids as assigned by motpy
    centroids: list of numpy arrays of centroid coordinates of bounding boxes drawn by detector
    frames: list of lists of frame numbers
    """

    def __init__(self, json_path: str = None):
        if json_path is None:
            self.track_id = []
            self.centroids = []
            self.frames = []
        else:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
            self.track_id = data["track_id"]
            self.centroids = [np.array([c[0], c[1]]) for c in data["centroids"]]
            self.frames = [np.array([f]) for f in data["frames"]]

    def update(self, active_tracks, img, current_frame):
        """
        Function to update ParticleTracks object with latest tracking results
        :param active_tracks: active_tracks attribute of motpy multiple object tracker
        :param img: OpenCV grayscale image
        :param current_frame: index of currently tracked frame
        :return:
        """
        for track in active_tracks:
            draw_track(img, track, thickness = 2)
            if track.id in self.track_id:
                idx = self.track_id.index(track.id)
                xy = np.array([np.mean([track.box[0], track.box[2]]), np.mean([track.box[1], track.box[3]])])
                self.frames[idx] = np.append(self.frames[idx], current_frame)
                self.centroids[idx] = np.array(
                    [np.append(self.centroids[idx][0, :], xy[0]),
                     np.append(self.centroids[idx][1, :], xy[1])])
            else:
                self.track_id.append(track.id)
                self.frames.append(np.array([current_frame]))
                self.centroids.append(
                    np.array([[np.mean([track.box[0], track.box[2]])], [np.mean([track.box[1], track.box[3]])]]))
        cv2.imshow('preview', img)
        cv2.waitKey(1)
        return self

    def filter_by_center_circle(self):
        """
        Predicts a center circle from the particle tracks by the following method:
        - Estimate first guess for center point as median of all tracks' points
        - Estimate radius as 1.5 times median distance of all points to estimated center
        - Estimate second guess for center point as median of all points inside first circle guess
        - Estimate radius as 0.95 quantile distance of all points inside first circle guess to 2nd guess center
        - Estimate actual center as mean of first and second guess
        - Estimate actual radius as mean of first and second guess
        :return:
        """
        xy = np.hstack(self.centroids)
        x_med = int(np.median(xy[0, :]))
        y_med = int(np.median(xy[1, :]))
        r = np.sqrt(np.square(xy[0, :] - x_med) + np.square(xy[1, :] - y_med) * 1.5)
        r_med = int(np.median(r) * 1.5)
        inner_ids = r < r_med
        xy_in = xy[:, inner_ids]
        x_med_in = int(np.median(xy_in[0, :]))
        y_med_in = int(np.median(xy_in[1, :]))
        r = np.sqrt(np.square(xy_in[0, :] - x_med_in) + np.square(xy_in[1, :] - y_med_in))
        r_in = int(np.quantile(r, 0.95))
        x_center = int((x_med + x_med_in) / 2.0)
        y_center = int((y_med + y_med_in) / 2.0)
        r_circle = int((r_med / 1.5 + r_in) / 2.0)
        for i in range(len(self.track_id)):
            r = np.sqrt(np.square(self.centroids[i][0, :] - x_center) + np.square(self.centroids[i][1, :] - y_center))
            outer_ids = r > r_circle
            self.centroids[i] = self.centroids[i][:, outer_ids]
            self.frames[i] = self.frames[i][outer_ids]
        return self

    def filter_by_length(self, min_length: int):
        """
        Function to filter tracks by track length as measured by number of frames
        """
        k = len(self.track_id) - 1
        while k >= 0:
            if len(self.frames[k]) < min_length:
                del self.frames[k]
                del self.centroids[k]
                del self.track_id[k]
            k -= 1
        return self

    def filter_by_velocity(self, min_velocity: float):
        """
        Function to filter tracks by minimum magnitude of velocity in px/frame
        """
        for k in range(len(self.track_id)):
            dx_df = np.gradient(self.centroids[k][0, :], self.frames[k])
            dy_df = np.gradient(self.centroids[k][1, :], self.frames[k])
            ds_df = np.sqrt(np.square(dx_df) + np.square(dy_df))
            keep_id = ds_df > min_velocity
            self.centroids[k] = self.centroids[k][:, keep_id]
            self.frames[k] = self.frames[k][keep_id]
        return self

    def filter_by_linearity(self, min_p: float):
        """
        Function to filter tracks by linearity as evaluated by pearson correlation coefficient
        """
        k = len(self.track_id) - 1
        while k >= 0:
            p_x = pearsonr(self.centroids[k][0, :], self.frames[k])[0]
            p_y = pearsonr(self.centroids[k][1, :], self.frames[k])[0]
            if abs(p_x) < min_p or abs(p_y) < min_p:
                del self.frames[k]
                del self.centroids[k]
                del self.track_id[k]
            k -= 1
        return self

    def filter_tracks(self, filter_criteria):
        """
        Function to filter tracks by various criteria.
        """
        self.filter_by_length(3)
        if filter_criteria.filter_by_velocity:
            self.filter_by_velocity(filter_criteria.min_velocity)
        if filter_criteria.filter_by_circle:
            self.filter_by_center_circle()
        if filter_criteria.filter_by_length:
            self.filter_by_length(filter_criteria.min_length)
        if filter_criteria.filter_by_linearity:
            self.filter_by_linearity(filter_criteria.min_p)
        return self

    def save_to_json(self, input_directory):
        data = {
            "track_id": self.track_id,
            "centroids": [c.tolist() for c in self.centroids],
            "frames": [f.tolist() for f in self.frames]
        }
        with open(input_directory + "\\particle_tracks.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

    class FilterCriteria(object):
        """
        Subclass to store filter criteria
        filter_by_length: flag to activate filter by length
        filter_by_velocity: flag to activate filter by velocity
        filter_by_linearity: flag to activate filter by linearity
        min_length: minimum track length
        min_velocity: minimum particle velocity in px/f
        min_p: minimum value of pearson correlation coefficient
        """

        def __init__(self, filter_by_length: bool = True, filter_by_circle: bool = True,
                     filter_by_velocity: bool = True, filter_by_linearity: bool = True,
                     min_length: int = 10, min_velocity: float = 5.0, min_p: float = 0.98):
            self.filter_by_length = filter_by_length
            self.filter_by_circle = filter_by_circle
            self.filter_by_velocity = filter_by_velocity
            self.filter_by_linearity = filter_by_linearity
            self.min_length = min_length
            self.min_velocity = min_velocity
            self.min_p = min_p
