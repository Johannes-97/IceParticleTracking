# IceParticleTracking
Motion-based multiple-object tracking of ice particles in image sequences
of propeller ice shedding. The track by detection workflow consists
of an object detector based on OpenCV's *backgroundSubtractorMOG2()* mixture of gaussian background subtractor
and a Kalman filter multiple-object tracker imported from motpy.

After creating a **Detector**, a **MultiObjectTracker**, and a **ParticleTrack** class object,
the main script *ice_particle_tracking.py* loops over all input images, performing the
following steps in each iteration:
* Read image and frame number
* Detect blobs by applying the detector's *detect()* method
* Update active tracks and assign new tracks through the tracker's *step()* method
* Save track location and frame number to **ParticleTracks** object using *update()* method

The strategy is to have very loose criteria during the detection and tracking processes
to collect a large quantity of data, filtering and cropping invalid tracks afterward by applying
 the *filter()* method of **ParticleTracks**. Filter criteria can be provided
in an object of the subclass **FilterCriteria**.

The resulting particle tracks are stored to a *particle_tracks.json* file in the input directory
and if desired, an image of the tracking result is shown.

## Detector
The detector class has one attribute *background_subtractor* and a method *detect()*,
which works by applying the background subtraction followed by a canny edge detection
and contour finding algorithm. Minimum enclosing circles are drawn around the contours to determine
the minimum enclosing radius, as well as the center point of each blob.
Only blobs with a radius within a specified range are considered as detections,
drawing a square bounding box of length 2*radius around the center point and creating
an object of motpy's **Detection** class for each detection.

## MultiObjectTracker
This part of the script is entirely taken from motpy. The following tracker
settings have proven successful:
```
model_spec = {
    'order_pos': 1, 'dim_pos': 2,
    'order_size': 0, 'dim_size': 2,
    'q_var_pos': 1.,
    'r_var_pos': 0.01
}
tracker = MultiObjectTracker(
    dt=0.1,
    model_spec=model_spec,
    matching_fn=IOUAndFeatureMatchingFunction(min_iou=0.01),
    active_tracks_kwargs={'min_steps_alive': 1, 'max_staleness': 1},
    tracker_kwargs={'max_staleness': 2})
```
## ParticleTracks
This class was created to store tracks' location and time (frame) data.
It has three basic attributes: *track_id*, *centroids*, *frames*.
The *update()* method iterates over all active tracks as assigned by the
tracker, stores the data to the **ParticleTracks** object and shows a preview image
of the tracking process.
Four filtering methods allow to filter the particle track data:
* Filter by estimating a rotor disk
* Filter by particle velocity
* Filter by track length
* Filter by linearity described by Pearson's correlation coefficient