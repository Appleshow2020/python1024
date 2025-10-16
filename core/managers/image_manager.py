import cv2
import time
from collections import deque
import sqlite3
import datetime
import os
from utils.config import ConfigManager
from utils.printing import printf, LT

# IMAGE_DIR = "C:\\Users\\zzuns\\Desktop\\Python1024\\python1024\\images"
# DB_DIR = "C:\\Users\\zzuns\\Desktop\\Python1024\\python1024\\db\\images.db"

class ImageManager:
    
    def __init__(self):
        self.IMAGE_DIR = ConfigManager.get_config().get("processing", None).get("image_dir", None)
        self.DB_DIR = ConfigManager.get_config().get("processing", None).get("db_dir", None)
        if self.IMAGE_DIR is None or self.DB_DIR is None:
            self.IMAGE_DIR = "C:\\Users\\zzuns\\Desktop\\Python1024\\python1024\\images"
            self.DB_DIR = "C:\\Users\\zzuns\\Desktop\\Python1024\\python1024\\db\\images.db"
            printf("Image directory or DB directory not configured properly, Using default directory:", 
                   f"IMAGE_DIR : {self.IMAGE_DIR}", 
                   f"DB_DIR : {self.DB_DIR}", ptype=LT.warning, sep = '\n')

    def init_db(self):
        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                timestamp DATETIME,
                filepath TEXT
            )
        """)
        conn.commit()
        conn.close()
        printf("Image Database initialized at ", self.DB_DIR, ptype=LT.info)
    
    def save_frame_and_record(self, camera_id, frame):
        timestamp = datetime.datetime.utcnow()
        filename = f"{camera_id}_{timestamp.strftime('%H%M%S_%f')}.jpg"
        filepath = os.path.join(self.IMAGE_DIR, filename)

        cv2.imwrite(filepath, frame)

        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO image_paths (camera_id, timestamp, filepath) VALUES (?, ?, ?)",
            (camera_id, timestamp.timestamp(), filepath)
        )
        conn.commit()
        conn.close()
        # printf(f"Saved frame from camera {camera_id} at {filepath}", ptype=LT.debug)
    
    def get_lastest_image_path(self, camera_id, threshold: int | None):
        if threshold is None or threshold <= 0:
            threshold = ConfigManager.get_config().get("processing", None).get("get_latest_image_paths_threshold", 5)
        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filepath FROM image_paths WHERE camera_id = ? ORDER BY timestamp DESC LIMIT ?",
            (camera_id, threshold,)
        )
        result = cursor.fetchone()
        conn.close()
        if not result:
            return None
        return result[0]
    
    def delete_old_images(self, time_minutes = 5):
        time_minutes = max(ConfigManager.get_config().get("processing", None).get("image_retention_minutes", 5), time_minutes)
        cutoff_time = time.time() - time_minutes * 60  # 5 minutes in seconds

        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute("SELECT id, filepath, timestamp FROM image_paths")
        rows = cursor.fetchall()

        deleted_count = 0
        for row in rows:
            img_id, filepath, timestamp = row
            if timestamp < cutoff_time:
                try:
                    os.remove(filepath)
                    cursor.execute("DELETE FROM image_paths WHERE id = ?", (img_id,))
                    deleted_count += 1
                    printf(f"Deleted old image: {filepath}", ptype=LT.debug)
                except Exception as e:
                    printf(f"Failed to delete image {filepath}: {e}", ptype=LT.error)

        conn.commit()
        conn.close()
        printf(f"Deleted {deleted_count} old images older than {time_minutes} minutes", ptype=LT.info)
