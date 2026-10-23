import numpy as np

def calculate_luminance(color_image, bbox):
    """
    Calculates the average luminance for the total image and a specific object bounding box.
    
    Parameters:
    - color_image: numpy array of shape (H, W, 3)
    - bbox: list or tuple of (xmin, ymin, xmax, ymax) defining the object boundaries.
    
    Returns:
    - luminance_total: float, average luminance of the entire image.
    - luminance_object: float, average luminance of the cropped object.
    """
    # 1. Extract color channels
    # ITU-R BT.601 weights: Y = 0.299*R + 0.587*G + 0.114*B
    # We use BGR!
    b = color_image[:, :, 0]
    g = color_image[:, :, 1]
    r = color_image[:, :, 2]
    
    # Calculate the luminance matrix for the total image
    luminance_matrix = 0.299 * r + 0.587 * g + 0.114 * b
    luminance_total = float(np.mean(luminance_matrix))
    
    # Handle the bounding box slice for the object
    try:
        x_min, y_min, x_max, y_max = bbox
        object_luminance_matrix = luminance_matrix[y_min:y_max, x_min:x_max]
        
        # Check if the bounding box has a valid size to avoid division-by-zero errors
        if object_luminance_matrix.size > 0:
            luminance_object = float(np.mean(object_luminance_matrix))
        else:
            luminance_object = None
            
    except (TypeError, ValueError, IndexError):
        # Fallback if the bbox format is corrupted, empty, or out of image bounds
        luminance_object = None
        
    return luminance_total, luminance_object