from typing import Dict, Generator, List, NamedTuple

from .image_stream import ImageStream
from listable import Listable


class Camera(Listable):
    name: str
    streams: Dict[str, ImageStream]

    def __init__(self, name: str, streams: List[ImageStream]) -> None:
        super().__init__()
        self.name = name
        self.streams = {stream.name: stream for stream in streams}

    def get_description(self) -> str:
        return self.name

    def enumerate_children(self) -> Generator[Listable, None, None]:
        yield from []
