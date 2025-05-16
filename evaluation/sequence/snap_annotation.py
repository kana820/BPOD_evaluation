from pathlib import Path
import json
from typing import Dict

from pydantic import BaseModel

from . import sequence as sq


class StreamAnnotation(BaseModel):
    start: int
    first_hand_appears: int
    first_hand_disappears: int


class CameraAnnotation(BaseModel):
    rgb: StreamAnnotation
    stereo: StreamAnnotation


class SnapAnnotation:
    sequence: "sq.Sequence"
    path: Path
    cameras: Dict[str, CameraAnnotation]

    def __init__(self, path: Path, sequence: "sq.Sequence"):
        self.path = path
        self.sequence = sequence
        with path.open("r") as f:
            annotation = json.load(f)
        self.cameras = {
            camera_name: CameraAnnotation(**camera_annotation)
            for camera_name, camera_annotation in annotation.items()
        }
