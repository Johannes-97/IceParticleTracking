import numpy as np
from scipy.stats import pearsonr


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
