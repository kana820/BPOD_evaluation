from pathlib import Path

from ..image_stream import ImageStream
from ..camera import Camera


def get_camera() -> Camera:
    base_path = Path.cwd() / "src/dataset/camera/d415"
    left_stream = ImageStream.load_from_files(
        "left",
        "stereo",
        1,
        base_path / "intrinsicCalibration_left.txt",
        base_path / "vignette_left.png",
        base_path / "pcalib_left.txt",
    )
    right_stream = ImageStream.load_from_files(
        "right",
        "stereo",
        1,
        base_path / "intrinsicCalibration_right.txt",
        base_path / "vignette_right.png",
        base_path / "pcalib_right.txt",
    )
    rgb_stream = ImageStream.load_from_files(
        "rgb",
        "rgb",
        3,
        base_path / "intrinsicCalibration_center.txt",
    )
    return Camera("d415", [left_stream, right_stream, rgb_stream])
