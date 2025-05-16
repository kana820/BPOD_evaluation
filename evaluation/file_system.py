import os
from typing import Generator
from pathlib import Path


def list_dir(root: str, return_folders: bool) -> Generator[str, None, None]:
    """Enumerate the folders or files inside the specified folder."""
    for thing in os.listdir(root):
        thing_path = os.path.join(root, thing)
        if os.path.isdir(thing_path) != return_folders:
            continue
        yield thing_path, thing

def listdir(dir: Path, files: bool = False, dirs: bool = False) -> Generator[Path, None, None]:
    """It's like the function above, but better."""
    assert dir.is_dir()
    for path in dir.iterdir():
        if files and path.is_file() or dirs and path.is_dir():
            yield path