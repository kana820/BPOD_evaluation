import functools
import dataclasses
from pathlib import Path

import numpy as np
import imageio


@dataclasses.dataclass(frozen=True, eq=True)
class Image:
    path: Path
    timestamp: float  # seconds since snap

    def get_data(self) -> np.ndarray:
        # Color images are RGB, not BGR.
        return imageio.imread(self.path)
