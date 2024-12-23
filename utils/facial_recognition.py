import threading
import cv2
import numpy as np
from deepface import DeepFace
import time
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime
import sqlite3
import pickle

class UserDatabase:
    def __init__(self, db_path: str = "face_recognition.db"):
        """Initialize the user database"""
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create the database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     email TEXT,
                     phone TEXT,
                     face_encoding BLOB,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Create recognition_logs table
        c.execute('''CREATE TABLE IF NOT EXISTS recognition_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id INTEGER,
                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     confidence REAL,
                     FOREIGN KEY (user_id) REFERENCES users(id))''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, name: str, face_encoding: Any, email: str = None, phone: str = None) -> int:
        """Add a new user to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        encoded_face = pickle.dumps(face_encoding)
        c.execute('''INSERT INTO users (name, email, phone, face_encoding)
                    VALUES (?, ?, ?, ?)''', (name, email, phone, encoded_face))
        
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        return user_id
    
    def get_all_users(self) -> List[Dict]:
        """Retrieve all users from the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT id, name, email, phone, face_encoding FROM users''')
        users = []
        for row in c.fetchall():
            users.append({
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'phone': row[3],
                'face_encoding': pickle.loads(row[4])
            })
        
        conn.close()
        return users
    
    def log_recognition(self, user_id: int, confidence: float):
        """Log a face recognition event"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO recognition_logs (user_id, confidence)
                    VALUES (?, ?)''', (user_id, confidence))
        
        conn.commit()
        conn.close()

class FacialAnalysisSystem:
    def __init__(self, db_path: str = "face_recognition.db"):
        """
        Initialize the facial analysis system
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.user_db = UserDatabase(db_path)
        self.analysis_results = {}
        self.lock = threading.Lock()
        self.recognized_users = {}
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def register_new_user(self, frame: np.ndarray, name: str, email: str = None, 
                         phone: str = None) -> bool:
        """
        Register a new user with their face encoding
        
        Args:
            frame: Image frame containing the user's face
            name: Name of the user
            email: Optional email address
            phone: Optional phone number
        Returns:
            Boolean indicating success
        """
        try:
            # Get face encoding using DeepFace
            embedding = DeepFace.represent(frame, enforce_detection=True)
            
            # Add user to database
            user_id = self.user_db.add_user(name, embedding, email, phone)
            print(f"Successfully registered user {name} with ID {user_id}")
            return True
        except Exception as e:
            print(f"Failed to register user: {str(e)}")
            return False
    
    def recognize_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Recognize a face in the frame
        
        Args:
            frame: Input image frame
        Returns:
            Dictionary containing user information if recognized
        """
        try:
            # Get face embedding
            embedding = DeepFace.represent(frame, enforce_detection=True)
            
            # Compare with all registered users
            users = self.user_db.get_all_users()
            for user in users:
                result = DeepFace.verify(
                    embedding,
                    user['face_encoding'],
                    distance_metric="cosine",
                    model_name="VGG-Face"
                )
                
                if result['verified']:
                    self.user_db.log_recognition(user['id'], result['distance'])
                    return user
            
            return None
        except Exception as e:
            print(f"Recognition error: {str(e)}")
            return None

    def analyze_face(self, frame: np.ndarray) -> Dict:
        """Analyze facial attributes"""
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=False
            )
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {}

    def find_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def draw_results(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                    analysis: Dict, recognized_user: Optional[Dict] = None) -> np.ndarray:
        """Draw analysis and recognition results on the frame"""
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Draw user information if recognized
        if recognized_user:
            y_pos = 30
            cv2.putText(frame, f"User: {recognized_user['name']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if recognized_user['email']:
                y_pos += 30
                cv2.putText(frame, f"Email: {recognized_user['email']}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if recognized_user['phone']:
                y_pos += 30
                cv2.putText(frame, f"Phone: {recognized_user['phone']}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw analysis results
        if analysis:
            y_pos = frame.shape[0] - 120
            cv2.putText(frame, f"Age: {analysis.get('age', 'N/A')}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y_pos += 30
            cv2.putText(frame, f"Emotion: {analysis.get('dominant_emotion', 'N/A')}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            y_pos += 30
            cv2.putText(frame, f"Gender: {analysis.get('gender', 'N/A')}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame

    def process_video(self, camera_index: int = 0) -> None:
        """Process video stream with face recognition"""
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        counter = 0
        analysis = {}
        recognized_user = None
        
        print("\nControls:")
        print("Press 'q' to quit")
        print("Press 'r' to register new user")
        print("Press 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            
            if ret:
                faces = self.find_faces(frame)
                
                # Update analysis and recognition every 30 frames
                if counter % 30 == 0 and len(faces) > 0:
                    try:
                        # Get the largest face for analysis
                        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                        face_frame = frame[y:y+h, x:x+w]
                        
                        threading.Thread(
                            target=lambda: analysis.update(self.analyze_face(face_frame))
                        ).start()
                        
                        threading.Thread(
                            target=lambda: setattr(self, 'recognized_user', 
                                                 self.recognize_face(face_frame))
                        ).start()
                    except Exception as e:
                        print(f"Processing error: {str(e)}")
                
                counter += 1
                
                # Draw results
                frame = self.draw_results(frame, faces, analysis, 
                                        getattr(self, 'recognized_user', None))
                cv2.imshow('Face Recognition', frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"recognition_{timestamp}.jpg", frame)
                    print(f"Saved frame as recognition_{timestamp}.jpg")
                elif key == ord('r'):
                    if len(faces) > 0:
                        name = input("Enter user name: ")
                        email = input("Enter email (optional): ") or None
                        phone = input("Enter phone (optional): ") or None
                        
                        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                        face_frame = frame[y:y+h, x:x+w]
                        
                        if self.register_new_user(face_frame, name, email, phone):
                            print("User registered successfully!")
                        else:
                            print("Failed to register user.")
                    else:
                        print("No face detected. Please try again.")
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage:
"""
# Initialize the system
recognition_system = FacialAnalysisSystem()

# Start video processing with face recognition
recognition_system.process_video(camera_index=0)
"""