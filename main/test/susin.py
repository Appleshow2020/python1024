import cv2
import time
from threading import Thread

from collections import deque
import sqlite3
from classes.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D
from classes.CameraCalibration import CameraCalibration
from classes.UserInterface import UserInterface
import datetime
import os

IMAGE_DIR = "C:\\Users\\User\\Desktop\\Python1024\\realpython1024\\python1024\\data\\image"
DB_DIR = "C:\\Users\\User\\Desktop\\Python1024\\realpython1024\\python1024\\data\\db\\images.db"

urls = {0: 'http://192.168.0.4:8000/video_feed',
        1: 'http://192.168.0.66:8000/video_feed',
        2: 'http://192.168.0.91:8000/video_feed'}
frame_queues = {cam_id: deque(maxlen=1) for cam_id in urls}
stop_flag = False

def init_db():
    conn = sqlite3.connect(DB_DIR)
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

def save_frame_and_record(camera_id, frame):
    timestamp = datetime.datetime.utcnow()
    filename = f"{camera_id}_{timestamp.strftime('%H%M%S_%f')}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)

    cv2.imwrite(filepath, frame)

    conn = sqlite3.connect(DB_DIR)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO image_paths (camera_id, timestamp, filepath) VALUES (?, ?, ?)",
        (camera_id, timestamp.timestamp(), filepath)
    )
    conn.commit()
    conn.close()

def load_camera_configs_from_txt(filepath):
    camera_configs = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue  # 빈 줄 무시
            parts = line.strip().split()
            if len(parts) != 7:
                raise ValueError(f"Invalid line format: {line}")
            
            cam_id = parts[0]
            pos = list(map(float, parts[1:4]))
            rot = list(map(float, parts[4:7]))
            camera_configs.append({
                "id": cam_id,
                "position": pos,
                "rotation": rot
            })
    return camera_configs

def BallPlaceChecker(bx: float, by: float): 
    # global Positioned
    # if ((bx <= 0)and(bx >= PointList[4][0])) and ((by <= PointList[4][1])and(by>=PointList[8][1])):
    #     Positioned[4] = True
    #     Positioned[5] = False
    #     Positioned[6] = False
    #     Positioned[7] = False
    #     return "li"
    
    # elif ((bx >= 0)and(bx <= PointList[7][0])) and ((by <= PointList[7][1])and(by>=PointList[11][1])):
    #     Positioned[4] = False
    #     Positioned[5] = True
    #     Positioned[6] = False
    #     Positioned[7] = False
    #     return "ri"
    
    # elif ((((bx >= PointList[2][0])and(bx <= PointList[3][0])) and ((by <= PointList[2][1])and(by >= PointList[6][1]))) or 
    #       (((bx >= PointList[7][0])and(bx <= PointList[3][0])) and ((by <= PointList[7][1])and(by >= PointList[11][1]))) or
    #       (((bx >= PointList[10][0])and(bx <= PointList[15][0])) and ((by <= PointList[10][1])and(by >= PointList[14][1])))):
    #     Positioned[4] = False
    #     Positioned[5] = False
    #     Positioned[6] = True
    #     Positioned[7] = False
    #     return "lo"
    
    # elif ((((bx >= PointList[0][0])and(bx <= PointList[1][0])) and ((by <= PointList[0][1])and(by >= PointList[5][1]))) or 
    #       (((bx >= PointList[0][0])and(bx <= PointList[4][0])) and ((by <= PointList[4][1])and(by >= PointList[8][1]))) or
    #       (((bx >= PointList[12][0])and(bx <= PointList[13][0])) and ((by <= PointList[8][1])and(by >= PointList[13][1])))):
    #     Positioned[4] = False
    #     Positioned[5] = False
    #     Positioned[6] = False
    #     Positioned[7] = True
    #     return "ro"
    
    # else:
    #     Positioned[4] = False
    #     Positioned[5] = False
    #     Positioned[6] = False
    #     Positioned[7] = False
    #     return None

    pass

def pointlistfunc():
    global PointList
    pdx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
    pdy = [7, 4, -4, -7]
    i=0
    while (len(PointList)!=16):
        PointList.append([pdx[i], pdy[i//4]])
        i+=1

def get_latest_image_paths():
    conn = sqlite3.connect(DB_DIR)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT camera_id, filepath FROM image_paths
        WHERE (camera_id, timestamp) IN (
            SELECT camera_id, MAX(timestamp)
            FROM image_paths
            GROUP BY camera_id
        )
    """)
    results = cursor.fetchall()
    conn.close()
    return results  # List[(cam_id, filepath)]

def camera_thread(cam_id, url):
    global stop_flag
    print(f"[{time.strftime('%X')}] [INFO] Starting camera thread {cam_id} with URL {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"\033[31m[{time.strftime('%X')}] [ERROR] Failed to open cam:{cam_id+1} at {url}\033[0m")
        return

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            frame_queues[cam_id].append(frame)
            save_frame_and_record(cam_id, frame)
        else:
            print(f"\033[31m[{time.strftime('%X')}] [ERROR] Failed to recept frame from cam:{cam_id} at {url}\033[0m")
        time.sleep(1 / 30)

    cap.release()

# 각 카메라마다 스레드 실행
threads = []
for cam_id, url in urls.items():
    t = Thread(target=camera_thread, args=(cam_id, url))
    t.start()
    threads.append(t)

vl={}
gl={}
pl={}

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out","Running"]
Positioned = [False]*len(Positioned)

# PointList = []
# pointlistfunc()
init_db()
camera_configs = load_camera_configs_from_txt("C:\\Users\\User\\Desktop\\Python1024\\realpython1024\\python1024\\test\\camera_configstest.txt")
calibrate = CameraCalibration(camera_configs,800,800,640,360)
camera_params = calibrate.get_camera_params()
calibrate.print_projection_matrices()
tracker = BallTracker3D(camera_params)
animate = Animation(vl, gl, pl)
interface = UserInterface()
animate.main()
interface.update(Positioned)

pts_2d = []
cam_ids = []

# 종료 처리
for t in threads:
    t.join()
cv2.destroyAllWindows()
