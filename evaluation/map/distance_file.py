from pathlib import Path
from typing import List, NamedTuple, Union


class Measurement(NamedTuple):
    from_point: str
    to_point: str
    distance: Union[float, str]


class DistanceFile:
    path: Path
    measurements: List[Measurement]

    def __init__(self, path: Path):
        self.path = path
        self.measurements = []

        # Read the measurements.
        with path.open("r") as f:
            lines = f.readlines()
        assert lines[0].strip() == "unit meters"
        for line in lines[1:]:
            from_point, to_point, distance = line.split()
            try:
                distance = float(distance)
            except:
                pass
            self.measurements.append(Measurement(from_point, to_point, distance))
