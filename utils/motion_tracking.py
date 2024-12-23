import numpy as np
import cv2

heatmap = None

def track_motion(frame):
    """Track motion and update heatmap."""
    global heatmap
    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    heatmap += motion_mask
    return motion_mask

def generate_heatmap(filename="heatmap.jpg"):
    """Save heatmap as an image."""
    global heatmap
    if heatmap is not None:
        cv2.imwrite(filename, heatmap)
