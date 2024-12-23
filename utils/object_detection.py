import torch
import cv2
import numpy as np
from PIL import Image
import time
from scipy.spatial import distance
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define classes to watch for and capture
WATCH_CLASSES = ['person', 'car']

def get_dominant_color(img, box):
    """
    Extract dominant color from the detected object
    
    Args:
        img: Input image
        box: Bounding box coordinates [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, box[:4])
    
    # Extract object region
    obj_region = img[y1:y2, x1:x2]
    
    # Convert to RGB for better color analysis
    obj_region_rgb = cv2.cvtColor(obj_region, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = obj_region_rgb.reshape(-1, 3)
    
    # Calculate the mean color
    mean_color = np.mean(pixels, axis=0)
    
    # Define color ranges
    color_ranges = {
        'red': ([150, 0, 0], [255, 50, 50]),
        'green': ([0, 150, 0], [50, 255, 50]),
        'blue': ([0, 0, 150], [50, 50, 255]),
        'white': ([200, 200, 200], [255, 255, 255]),
        'black': ([0, 0, 0], [50, 50, 50]),
        'yellow': ([200, 200, 0], [255, 255, 50]),
        'purple': ([150, 0, 150], [255, 50, 255]),
        'orange': ([200, 100, 0], [255, 150, 50])
    }
    
    # Find the closest color
    min_dist = float('inf')
    dominant_color = 'unknown'
    
    for color_name, (lower, upper) in color_ranges.items():
        dist = distance.euclidean(mean_color, np.mean([lower, upper], axis=0))
        if dist < min_dist:
            min_dist = dist
            dominant_color = color_name
    
    return dominant_color

def process_frame(frame, conf_threshold=0.25, last_capture_time=None, min_capture_interval=2.0):
    """
    Process a single frame with YOLOv5 and capture images of watched objects
    
    Args:
        frame: Input frame from webcam or image
        conf_threshold: Confidence threshold for detections
        last_capture_time: Dictionary tracking last capture time for each class
        min_capture_interval: Minimum time (in seconds) between captures of the same class
    """
    if last_capture_time is None:
        last_capture_time = {cls: 0 for cls in WATCH_CLASSES}
        
    # Create captures directory if it doesn't exist
    os.makedirs('captures', exist_ok=True)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(frame_rgb)
    
    # Get detections
    detections = results.xyxy[0].cpu().numpy()
    
    current_time = time.time()
    captured_classes = set()
    
    # Draw detections and handle captures
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        
        if conf >= conf_threshold:
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get object color
            color_name = get_dominant_color(frame, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            class_id = int(cls)
            class_name = model.names[class_id] if class_id < len(model.names) else str(class_id)
            label = f"{class_name} ({color_name}): {conf:.2f}"
            
            # Check if this is a watched class and enough time has passed since last capture
            if (class_name in WATCH_CLASSES and 
                current_time - last_capture_time[class_name] >= min_capture_interval and
                class_name not in captured_classes):
                
                # Save the frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"captures/{class_name}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Captured {class_name} at {timestamp}")
                
                # Update last capture time and add to captured classes
                last_capture_time[class_name] = current_time
                captured_classes.add(class_name)
            
            # Calculate text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1]
            
            # Draw label background
            cv2.rectangle(frame, (x1, text_y - text_size[1] - 5),
                         (x1 + text_size[0], text_y + 5), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame, last_capture_time

def detect_webcam(camera_index=0, conf_threshold=0.25):
    """
    Run YOLOv5 detection on webcam feed with automatic capture of watched objects
    
    Args:
        camera_index: Index of the webcam to use (default is 0 for primary webcam)
        conf_threshold: Confidence threshold for detections
    """
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Webcam: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Watching for classes: {', '.join(WATCH_CLASSES)}")
    
    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    # Initialize last capture time dictionary
    last_capture_time = {cls: 0 for cls in WATCH_CLASSES}
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Process frame
            processed_frame, last_capture_time = process_frame(
                frame, 
                conf_threshold=conf_threshold,
                last_capture_time=last_capture_time
            )
            
            # Calculate and display FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Add FPS counter to frame
            cv2.putText(processed_frame, f"FPS: {fps_display}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("YOLOv5 Webcam Detection", processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('s'):  # Press 's' to save frame manually
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"captures/manual_{timestamp}.jpg", processed_frame)
                print(f"Saved frame as manual_{timestamp}.jpg")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

# Example usage:
detect_webcam(camera_index=0, conf_threshold=0.25)