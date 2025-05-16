from .camera import Camera
from .d455 import get_camera as get_camera_d455
from .d415 import get_camera as get_camera_d415

# Add cameras here so that they can be accessed by name.
cameras = [
    get_camera_d455(),
    get_camera_d415(),
]
camera_map = {c.name: c for c in cameras}


def get_camera_by_name(name: str) -> Camera:
    return camera_map[name]