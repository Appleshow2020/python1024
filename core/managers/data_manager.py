from utils.printing import printf, LT
import time
import datetime
import os
from utils.config import ConfigManager
import sqlite3

class DataManager:
    def __init__(self):
        self.config = ConfigManager().get_config()
        self.DB_DIR = self.config.get("processing", None).get("db_dir", None)
        if self.DB_DIR is None:
            self.DB_DIR = "C:\\Users\\User\\Desktop\\Python1024\\python1024\\db\\data.db"
            printf("Data DB directory not configured properly, Using default directory:", 
                   f"DB_DIR : {self.DB_DIR}", ptype=LT.warning)

    def init_db(self):
        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                position_x REAL
                position_y REAL,
                position_z REAL,
                velocity_x REAL,
                velocity_y REAL,
                velocity_z REAL,
                direction_x REAL,
                direction_y REAL,
                direction_z REAL,
                zone_info TEXT,
                detection_count INTEGER
            )
        """)
        conn.commit()
        conn.close()
        printf("Tracking Database initialized at ", self.DB_DIR, ptype=LT.info)
    
    def save_tracking_data(self, tracking_result):
        conn = sqlite3.connect(self.DB_DIR)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tracking_data (timestamp, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, direction_x, direction_y, direction_z, zone_info, detection_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tracking_result.get('timestamp', datetime.datetime.now()),
            tracking_result.get('position', (0, 0, 0))[0],
            tracking_result.get('position', (0, 0, 0))[1],
            tracking_result.get('position', (0, 0, 0))[2],
            tracking_result.get('velocity', (0, 0, 0))[0],
            tracking_result.get('velocity', (0, 0, 0))[1],
            tracking_result.get('velocity', (0, 0, 0))[2],
            tracking_result.get('direction', (0, 0, 0))[0],
            tracking_result.get('direction', (0, 0, 0))[1],
            tracking_result.get('direction', (0, 0, 0))[2],
            tracking_result.get('zone_info', ''),
            tracking_result.get('detection_count', 0)
        ))
        conn.commit()
        conn.close()