import json
from pathlib import Path
from typing import Dict, Generator, List, Optional

from tqdm import tqdm
from pydantic.error_wrappers import ValidationError

from .marker_annotation import MarkerAnnotation
from .snap_annotation import SnapAnnotation
from camera.image_stream import ImageStream
from ...utils import get_icon
from ...evaluation.annotation_validation import validate_frame_boundaries
from ..camera import get_camera_by_name
from ..camera.camera import Camera
from ..camera.image import Image
from ..listable import Listable
from ...file_system import listdir
from .. import location as ln
from ...colors import cyan


class Sequence(Listable):
    raw_path: Path
    location: "ln.Location"
    cameras: Dict[str, Camera]
    marker_annotation: Optional[MarkerAnnotation]
    snap_annotation: Optional[SnapAnnotation]

    # Use get_images instead of accessing loaded_images directly.
    loaded_images: Dict[Camera, Dict[ImageStream, List[Image]]]

    def __init__(self, raw_path: Path, location: "ln.Location"):
        super().__init__()
        self.raw_path = raw_path
        self.location = location
        self.cameras = {}
        self.loaded_images = {}

        # Load the marker annotation.
        try:
            self.marker_annotation = MarkerAnnotation(
                self.raw_path / "3rd_person.json", self
            )
        except FileNotFoundError:
            self.marker_annotation = None

        # Load the snap annotation.
        try:
            self.snap_annotation = SnapAnnotation(
                self.raw_path / "frame_boundaries.json", self
            )
        except (FileNotFoundError, ValidationError):
            self.snap_annotation = None

        # First, try to get the cameras from the .bag files' names.
        found_cameras = False
        for file in listdir(raw_path, files=True):
            if file.suffix == ".bag":
                self.cameras[file.stem] = get_camera_by_name(file.stem)
                found_cameras = True

        # If that fails, get cameras from frame_boundaries.json.
        if not found_cameras:
            for file in listdir(raw_path, files=True):
                if file.name == "frame_boundaries.json":
                    with file.open("r") as f:
                        boundaries = json.load(f)
                    for camera_name in boundaries.keys():
                        self.cameras[camera_name] = get_camera_by_name(camera_name)
                break

    @property
    def processed_path(self) -> Path:
        return self.root_path / "processed" / self.location.name / self.name

    @property
    def is_backward(self) -> bool:
        is_forward = "forward" in self.raw_path.stem
        is_backward = "backward" in self.raw_path.stem
        assert is_forward != is_backward
        return is_backward

    @property
    def name(self) -> str:
        return self.raw_path.name

    @property
    def root_path(self) -> Path:
        return self.raw_path.parents[3]

    def get_description(self) -> str:
        checks = {
            "SE": self.snap_annotation is not None,  # Snap Exists
            "ME": self.marker_annotation is not None,  # Markers Exist
            "MV": self.marker_annotation is not None
            and self.location.map.is_annotation_consistent(
                self.marker_annotation, self.is_backward
            ),  # Markers Valid
        }
        check_string = ", ".join(
            [f"{key}: {get_icon(check)}" for key, check in checks.items()]
        )

        return f"{self.name} [{check_string}]"

    def enumerate_children(self) -> Generator[Listable, None, None]:
        for camera in self.cameras.values():
            yield camera

    def get_image_paths(self, camera: Camera, image_stream: ImageStream) -> List[Path]:
        """Don't use this anymore. Instead, use self.get_images(camera, image_stream)."""
        return [image.path for image in self.get_images(camera, image_stream)]

    def get_images(self, camera: Camera, image_stream: ImageStream) -> List[Image]:
        """Get a list of images for the specified camera and image stream. The images
        are ordered by timestamp.
        """

        # Load the images if they haven't been loaded.
        if (
            camera not in self.loaded_images
            or image_stream not in self.loaded_images[camera]
        ):
            self.load_images(camera, image_stream)

        # Return the loaded (cached) images.
        return self.loaded_images[camera][image_stream]

    def get_images_after_snap(
        self, camera: Camera, image_stream: ImageStream, buffer_seconds: float = 1.5
    ) -> List[Image]:
        images = self.get_images(camera, image_stream)
        return [image for image in images if image.timestamp > buffer_seconds]

    def load_images(self, camera: Camera, image_stream: ImageStream):
        print(cyan(f"Loading images for {camera.name}/{image_stream.name}."))

        # Get the snap frame.
        snap_frame_microseconds: int = getattr(
            self.snap_annotation.cameras[camera.name],
            image_stream.snap_annotation_name,
        ).start

        # Make image objects.
        stream_images = []
        image_directory = self.processed_path / camera.name / image_stream.name
        for image_path in tqdm(list(image_directory.iterdir())):
            if image_path.suffix != ".png" or not image_path.is_file():
                continue

            image_timestamp_microseconds = int(image_path.stem)
            image_timestamp_seconds = (
                image_timestamp_microseconds - snap_frame_microseconds
            ) / 1e6

            stream_images.append(
                Image(
                    image_path,
                    image_timestamp_seconds,
                )
            )

        ordered_images = sorted(stream_images, key=lambda image: image.timestamp)

        # Save the results to self.loaded_images.
        if camera not in self.loaded_images:
            self.loaded_images[camera] = {}
        self.loaded_images[camera][image_stream] = ordered_images
