import sqlite3
from config import DATABASE_PATH

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Activity log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Known faces table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    """)
    
    # Registered objects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registered_objects (
            id INTEGER PRIMARY KEY,
            object_name TEXT NOT NULL,
            image BLOB NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def add_known_face(name, encoding):
    """Add a new known face to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO known_faces (name, encoding) VALUES (?, ?)", (name, encoding))
    conn.commit()
    conn.close()

def get_known_faces():
    """Retrieve all known faces and their encodings."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM known_faces")
    faces = cursor.fetchall()
    conn.close()
    return [{"name": name, "encoding": encoding} for name, encoding in faces]

def add_registered_object(object_name, image):
    """Add a new registered object to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO registered_objects (object_name, image) VALUES (?, ?)", (object_name, image))
    conn.commit()
    conn.close()

def get_registered_objects():
    """Retrieve all registered objects."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT object_name FROM registered_objects")
    objects = [row[0] for row in cursor.fetchall()]
    conn.close()
    return objects
