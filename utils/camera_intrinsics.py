import json

def load_camera_intrinsics(file_path="camera_intrinsics.json"):
    """
    Reads a JSON file containing camera intrinsics and returns them as a dictionary.

    Returns:
        dict: {
            'fx': float,
            'fy': float,
            'ppx': float,
            'ppy': float,
            'distortion': list of floats
        }
    """
    try:
        with open(file_path, "r") as f:
            intrinsics = json.load(f)
        return intrinsics
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None