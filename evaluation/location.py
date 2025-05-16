from typing import List, Generator, Optional
from pathlib import Path

from sequence.sequence import Sequence
from ..file_system import listdir
from .map.map import Map
from .listable import Listable
from ..colors import gray, green, red


class Location(Listable):
    path: Path
    sequences: List[Sequence]
    map: Optional[Map]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        try:
            self.map = Map(path, self)
        except:
            self.map = None
        try:
            self.sequences = [
                Sequence(sequence, self)
                for sequence in listdir(path / "sequences", dirs=True)
            ]
        except:
            self.sequences = []

    @property
    def name(self) -> str:
        return self.path.name

    def get_description(self) -> str:
        color_function = green if len(self.sequences) == 4 else red
        return f"{color_function(self.path.name)} {gray(f'({len(self.sequences)} sequences)')}"

    def enumerate_children(self) -> Generator[Listable, None, None]:
        for sequence in self.sequences:
            yield sequence
