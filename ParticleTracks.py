import numpy as np
from scipy.stats import pearsonr
from motpy.testing_viz import draw_track
import cv2


class ParticleTracks(object):
    """
    Class to store particle track data
    track_id: list of track ids as assigned by motpy
    centroids: list of numpy arrays of centroid coordinates of bounding boxes drawn by detector
    frames: list of lists of frame numbers
    """

    def __init__(self):
        self.track_id = []
        self.centroids = []
        self.frames = []

    def update(self, active_tracks, img, current_frame):
        """
        Function to update ParticleTracks object with latest tracking results
        :param active_tracks: active_tracks attribute of motpy multiple object tracker
        :param img: OpenCV grayscale image
        :param current_frame: index of currently tracked frame
        :return:
        """
        for track in active_tracks:
            draw_track(img, track)
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

    def delete_stalled_tracks(self, active_tracks, current_frame, min_velocity):
        for track in active_tracks:
            if track.id in self.track_id:
                idx = self.track_id.index(track.id)
                xy = np.array([np.mean([track.box[0], track.box[2]]), np.mean([track.box[1], track.box[3]])])
                ds = np.linalg.norm(self.centroids[idx][:, -1] - xy)
                df = current_frame - self.frames[idx][-1]
                if ds / df < min_velocity:
                    del track
        return active_tracks

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
        k = len(self.track_id) - 1
        while k >= 0:
            dx_df = np.gradient(self.centroids[k][0, :], self.frames[k])
            dy_df = np.gradient(self.centroids[k][1, :], self.frames[k])
            v_min = np.min(np.sqrt(dx_df ** 2 + dy_df ** 2))
            if v_min < min_velocity:
                del self.frames[k]
                del self.centroids[k]
                del self.track_id[k]
            k -= 1
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
        if filter_criteria.filter_by_length:
            self.filter_by_length(filter_criteria.min_length)
        if filter_criteria.filter_by_velocity:
            self.filter_by_velocity(filter_criteria.min_velocity)
        if filter_criteria.filter_by_linearity:
            self.filter_by_linearity(filter_criteria.min_p)
        return self

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

        def __init__(self, filter_by_length: bool = True, filter_by_velocity: bool = True,
                     filter_by_linearity: bool = True, min_length: int = 10,
                     min_velocity: float = 5.0, min_p: float = 0.98):
            self.filter_by_length = filter_by_length
            self.filter_by_velocity = filter_by_velocity
            self.filter_by_linearity = filter_by_linearity
            self.min_length = min_length
            self.min_velocity = min_velocity
            self.min_p = min_p
