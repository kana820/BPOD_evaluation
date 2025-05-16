from pathlib import Path
import json
from typing import List, Optional
from dataclasses import dataclass

from . import sequence as sq


@dataclass
class Marker:
    timestamp: float  # seconds
    is_keypoint: bool


class MarkerAnnotation:
    sequence: "sq.Sequence"
    path: Path
    video_start_time: float
    markers: List[Marker]

    def __init__(
        self,
        path: Path,
        sequence: "sq.Sequence",
        frame_rate_hz: float = 30.0,
    ):
        self.path = path
        self.sequence = sequence
        with path.open("r") as f:
            annotation = json.load(f)

        # TODO: Make the snap frame key in the annotation JSON consistent so this isn't
        # necessary anymore.
        for key_name in ["snap_frame", "start_snap_frame"]:
            if key_name in annotation:
                start_snap_frame = annotation[key_name]
                break
        else:
            raise Exception("Starting frame not found!")
        
        self.video_start_time = -start_snap_frame / frame_rate_hz

        self.markers = []
        for marker in annotation["markers"]:
            timestamp = (marker["frame"] - start_snap_frame) / frame_rate_hz
            self.markers.append(Marker(timestamp, marker["type"] == "keyframe"))
