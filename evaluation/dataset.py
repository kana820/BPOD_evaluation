from pathlib import Path
from typing import Generator, List

from tqdm import tqdm

from location import Location
from file_system import listdir
from listable import Listable
from colors import cyan


class Dataset(Listable):
    locations: List[Location]

    def __init__(self, path: Path) -> None:
        super().__init__()
        print(cyan("Initializing dataset object."))
        self.locations = [
            Location(location)
            for location in tqdm(list(listdir(path / "raw", dirs=True)))
        ]

    def get_description(self) -> str:
        return "BPOD Dataset"

    def enumerate_children(self) -> Generator["Listable", None, None]:
        for location in self.locations:
            yield location
