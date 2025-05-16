from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import imageio

from .calibration import Calibration
from utils import is_numpy, is_shape

Resolution = Tuple[int, int, int]


class ImageStream:
    name: str
    snap_annotation_name: str
    _calibration: Optional[Calibration]
    _resolution: Optional[Resolution]
    _vignette: Optional[np.ndarray]
    _photometric: Optional[np.ndarray]

    def __init__(
        self,
        name: str,
        snap_annotation_name: str,
        calibration: Optional[Calibration] = None,
        resolution: Optional[Resolution] = None,
        vignette: Optional[np.ndarray] = None,
        photometric: Optional[np.ndarray] = None,
    ) -> None:
        self.name = name
        self.snap_annotation_name = snap_annotation_name
        self._calibration = calibration
        self._resolution = resolution
        self._vignette = vignette
        self._photometric = photometric

    @property
    def calibration(self) -> Calibration:
        """Return the intrinsic calibration matrix K."""
        assert isinstance(self._calibration, Calibration)
        return self._calibration

    @property
    def resolution(self) -> Resolution:
        """Return the resolution as (height, width, nchannels)."""
        assert is_shape(self._resolution, ndim=3)
        return self._resolution

    @property
    def vignette(self) -> np.ndarray:
        """Return the vignette. The resolution matches the camera stream's."""
        assert is_numpy(self._vignette, shape=self.resolution)
        return self._vignette

    @property
    def photometric(self) -> np.ndarray:
        """Return the photometric calibration. This remaps [0, 255] for each channel."""
        assert is_numpy(
            self._photometric, shape=(self.resolution[2], 256), dtype=np.float64
        )
        return self._photometric

    @staticmethod
    def load_instrinsic_calibration(
        path: Optional[Path], num_channels: int
    ) -> Tuple[Optional[Calibration], Optional[Resolution]]:
        if path is None:
            return None, None

        calibration = path.read_text().split("\n")

        # Read the focal length.
        focal_length, fx, fy = calibration[0].split()
        assert focal_length == "Focal_Length"
        fx = float(fx)
        fy = float(fy)

        # Read the principal point (note the typo in the format).
        principal_point, cx, cy = calibration[1].split()
        assert (
            principal_point == "Principle_Point" or principal_point == "Principal_Point"
        )
        cx = float(cx)
        cy = float(cy)

        # Read the radial distortion.
        radial_distortion, k1, k2 = calibration[2].split()
        assert radial_distortion == "Radial_Distortion"
        k1 = float(k1)
        k2 = float(k2)

        # Read the tangential distortion.
        tangential_distortion, p1, p2 = calibration[3].split()
        assert tangential_distortion == "Tangential_Distortion"
        p1 = float(p1)
        p2 = float(p2)

        # Read the resolution.
        size, width, height = calibration[4].split()
        assert size == "Size"
        width = int(width)
        height = int(height)

        return Calibration(fx, fy, cx, cy, k1, k2, p1, p2, width, height), (
            height,
            width,
            num_channels,
        )

    @staticmethod
    def load_vignette(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        vignette = imageio.imread(path)
        return vignette if vignette.ndim == 3 else vignette[:, :, None]

    @staticmethod
    def load_photometric(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        photometric = np.loadtxt(path, dtype=np.float64)[None, :]
        assert photometric.shape == (1, 256)
        return photometric

    @staticmethod
    def load_from_files(
        name: str,
        snap_annotation_name: str,
        num_channels: int,
        intrinsic_calibration: Optional[Path] = None,
        vignette: Optional[Path] = None,
        photometric: Optional[Path] = None,
    ) -> "ImageStream":
        return ImageStream(
            name,
            snap_annotation_name,
            *ImageStream.load_instrinsic_calibration(
                intrinsic_calibration, num_channels
            ),
            ImageStream.load_vignette(vignette),
            ImageStream.load_photometric(photometric),
        )
