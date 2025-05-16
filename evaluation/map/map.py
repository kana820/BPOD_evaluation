from pathlib import Path
from typing import Tuple, Union

from scipy import interpolate
import numpy as np
import json

from .. import location as ln
from ..sequence.marker_annotation import MarkerAnnotation
from .distance_file import DistanceFile


class Map:
    location: "ln.Location"
    points: np.ndarray
    is_keypoint: np.ndarray
    distance_file: DistanceFile

    def __init__(self, path: Path, location: "ln.Location"):
        self.points = np.load(path / "points.npy")
        self.location = location
        with (path / "point_types.json").open() as f:
            point_types = json.load(f)
        self.is_keypoint = np.array(point_types) == "keypoint"
        self.distance_file = DistanceFile(path / "distances.map")
        assert self.is_keypoint.shape[0] == self.points.shape[0]

    def is_annotation_consistent(
        self,
        annotation: MarkerAnnotation,
        backward: bool,
    ) -> bool:
        # If this condition isn't met, the annotations don't correspond to n full loops
        # around the map.
        if len(annotation.markers) % len(self.points) != 1:
            return False

        # Check each marker's consistency.
        for index, marker in enumerate(annotation.markers):
            map_index = (-index if backward else index) % len(self.points)
            if marker.is_keypoint != self.is_keypoint[map_index]:
                return False

        return True

    def sample(
        self,
        times: np.ndarray,
        annotation: MarkerAnnotation,
        backward: bool,
    ) -> np.ndarray:
        """Sample the ground-truth location at the specified times."""
        assert self.is_annotation_consistent(annotation, backward)

        marker_times = [marker.timestamp for marker in annotation.markers]
        marker_positions = [
            self.points[(-index if backward else index) % len(self.points)]
            for index, _ in enumerate(annotation.markers)
        ]
        f = interpolate.interp1d(
            marker_times,
            marker_positions,
            axis=0,
            bounds_error=False,
            fill_value=(marker_positions[0], marker_positions[-1]),
        )
        return f(times)

    def get_ground_truth(
        self,
        annotation: MarkerAnnotation,
        backward: bool,
        include_is_keypoint: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        marker_times = [marker.timestamp for marker in annotation.markers]
        marker_positions = [
            self.points[(-index if backward else index) % len(self.points)]
            for index, _ in enumerate(annotation.markers)
        ]
        result = np.concatenate(
            [np.array(marker_times)[:, None], np.array(marker_positions)],
            axis=1,
        )
        if include_is_keypoint:
            keypoints = [
                self.is_keypoint[(-index if backward else index) % len(self.points)]
                for index, _ in enumerate(annotation.markers)
            ]
            return result, keypoints
        return result
