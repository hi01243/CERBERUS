"""Runtime mutable shared state across Cerberus modules."""
import threading

CURRENT_SESSION = "SESSION_ROOM_A1"
monitoring_active = False

camera_registry = {}
registry_lock = threading.Lock()
_next_camera_num = 1


def generate_camera_id():
    global _next_camera_num
    with registry_lock:
        cam_id = f"cam{_next_camera_num}"
        _next_camera_num += 1
    return cam_id

