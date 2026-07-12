from core.services.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D
from classes.CameraCalibration import CameraCalibration
from core.managers.user_interface import UserInterface
import time
import cv2
import os
import sqlite3
import datetime
IMAGE_DIR = "C:\\Users\\User\\Desktop\\Python1024\\realpython1024\\python1024\\data\\image"
DB_DIR = "C:\\Users\\User\\Desktop\\Python1024\\realpython1024\\python1024\\data\\db\\images.db"
os.makedirs(IMAGE_DIR, exist_ok=True)

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

    # 이미지 저장
    cv2.imwrite(filepath, frame)

    # DB에 경로 저장
    conn = sqlite3.connect(DB_DIR)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO image_paths (camera_id, timestamp, filepath) VALUES (?, ?, ?)",
        (camera_id, timestamp, filepath)
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

camera_configs = load_camera_configs_from_txt("python1024\camera_configs.txt")
calibrate = CameraCalibration(camera_configs,800,800,640,360)
camera_params = calibrate.get_camera_params()
calibrate.print_projection_matrices()
tracker = BallTracker3D(camera_params)

vl={}
gl={}
pl={}

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out","Running"]
Positioned = [False]*len(Positioned)

def BallPlaceChecker(bx: float, by: float) -> str: 
    global Positioned
    if ((bx <= 0)and(bx >= PointList[4][0])) and ((by <= PointList[4][1])and(by>=PointList[8][1])):
        Positioned[4] = True
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = False
        return "li"
    
    elif ((bx >= 0)and(bx <= PointList[7][0])) and ((by <= PointList[7][1])and(by>=PointList[11][1])):
        Positioned[4] = False
        Positioned[5] = True
        Positioned[6] = False
        Positioned[7] = False
        return "ri"
    
    elif ((((bx >= PointList[2][0])and(bx <= PointList[3][0])) and ((by <= PointList[2][1])and(by >= PointList[6][1]))) or 
          (((bx >= PointList[7][0])and(bx <= PointList[3][0])) and ((by <= PointList[7][1])and(by >= PointList[11][1]))) or
          (((bx >= PointList[10][0])and(bx <= PointList[15][0])) and ((by <= PointList[10][1])and(by >= PointList[14][1])))):
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = True
        Positioned[7] = False
        return "lo"
    
    elif ((((bx >= PointList[0][0])and(bx <= PointList[1][0])) and ((by <= PointList[0][1])and(by >= PointList[5][1]))) or 
          (((bx >= PointList[0][0])and(bx <= PointList[4][0])) and ((by <= PointList[4][1])and(by >= PointList[8][1]))) or
          (((bx >= PointList[12][0])and(bx <= PointList[13][0])) and ((by <= PointList[8][1])and(by >= PointList[13][1])))):
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = True
        return "ro"
    
    else:
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = False
        return None


PointList = []
def pointlistfunc():
    global PointList
    pdx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
    pdy = [7, 4, -4, -7]
    i=0
    while (len(PointList)!=16):
        PointList.append([pdx[i], pdy[i//4]])
        i+=1

animate = Animation(vl, gl, pl)
interface = UserInterface()
interface.plot_table(Positioned)
frames = {}


pts_2d = []
cam_ids = range(4)
init_db()
caps = {cam_id: cv2.VideoCapture(0, cv2.CAP_DSHOW) for cam_id in cam_ids}
while True:
    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            save_frame_and_record(0, frame)
            frames[cam_id] = frame
        else:
            print(f"\033[31m[{time.strftime('%X')}] [ERROR] Failed to read from camera {cam_id}\033[0m")



    # DB에서 최근 이미지 경로 불러오기
    image_paths = get_latest_image_paths()

    for cam_id, filepath in image_paths:
        pt = tracker.detect_ball(filepath)
        if pt is not None:
            pts_2d.append(pt)
            cam_ids.append(int(cam_id))

    # 삼각측량
    if len(pts_2d) >= 2:
        position_3d = tracker.triangulate_point(pts_2d, cam_ids)
        timestamp = time.time()
        state = tracker.update_state(position_3d, timestamp)
        print(state)
        vl[timestamp] = state["velocity"]
        gl[timestamp] = state["direction"]
        pl[timestamp] = state["position"]
        BallPlaceChecker(state["position"][0], state["position"][1])
    else:
        print(f"\033[31m[{time.strftime('%X')}] [ERROR] Not enough camera data for triangulation.\033[0m")
    time.sleep(1/30)
