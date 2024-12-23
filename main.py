import cv2
from utils.facial_recognition import register_new_face, identify_faces, load_known_faces
from utils.object_detection import register_new_object, detect_objects
from utils.motion_tracking import track_motion, generate_heatmap
from utils.db_manager import init_db

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def display_menu():
    """Display the main menu."""
    print("\n=== Smart Home Security System ===")
    print("1. Add a Person")
    print("2. Add an Object")
    print("3. Detect Objects")
    print("4. Detect Persons (Facial Recognition)")
    print("5. Track Motion")
    print("6. Generate Heatmap")
    print("0. Exit")

def add_person():
    """Add a new person to the database."""
    print("\n--- Add a Person ---")
    register_new_face()

def add_object():
    """Add a new object to the database."""
    print("\n--- Add an Object ---")
    register_new_object()

def detect_objects_in_frame(camera_index=0):
    """Detect objects in a live camera feed."""
    print("\n--- Detecting Objects ---")
    cap = cv2.VideoCapture(camera_index)
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break
        
        detections = detect_objects(frame)
        for obj in detections["all_objects"]:
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            print(label)
        
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_persons_in_frame(camera_index=0):
    """Detect persons using facial recognition in a live camera feed."""
    print("\n--- Detecting Persons ---")
    cap = cv2.VideoCapture(camera_index)
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        identities, locations = identify_faces(frame)
        for identity, loc in zip(identities, locations):
            print(f"Detected: {identity}")
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, identity, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Facial Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def track_motion_in_frame(camera_index=0):
    """Track motion in a live camera feed."""
    print("\n--- Tracking Motion ---")
    cap = cv2.VideoCapture(camera_index)
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break
        
        motion_mask = track_motion(frame)
        cv2.imshow("Motion Tracking", motion_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run the system."""
    # Initialize the database and load known faces
    init_db()
    load_known_faces()
    
    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            add_person()
        elif choice == "2":
            add_object()
        elif choice == "3":
            detect_objects_in_frame()
        elif choice == "4":
            detect_persons_in_frame()
        elif choice == "5":
            track_motion_in_frame()
        elif choice == "6":
            print("\n--- Generating Heatmap ---")
            generate_heatmap()
            print("Heatmap saved as 'heatmap.jpg'.")
        elif choice == "0":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
