import cv2
import time
from threading import Thread
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, Optional, Tuple, List

from classes.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D as BallTracker3D
from classes.CameraCalibration import CameraCalibration
from classes.UserInterface import UserInterface
from classes.CameraPOCalc import CameraPOCalc





camera_count: int = int(input("Camera Count(1 int):"))
camera_indices: Dict[int, int] = {key: key+1 for key in range(camera_count)}

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 60
@dataclass
class CamStream:
    cap: Optional[cv2.VideoCapture]
    frames: Deque  

def now() -> float:
    
    return time.perf_counter()

def load_camera_configs():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    camera_configs = []
    if len(camera_cfgs.values()) == 7:
        cfg = {
            0: [6,6,(4,0,0),0]
        }
    for idx,i in camera_cfgs:
        w,h,alpha_h,alpha_v = i
        calculator = CameraPOCalc(w,h,alpha_h,alpha_v)
        calculator.solve()
    
def build_point_grid() -> List[Tuple[float, float]]:
    
    pdx = [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11]
    pdy = [7, 4, -4, -7]
    pts = []
    i = 0
    while len(pts) != 16:
        pts.append((pdx[i], pdy[i // 4]))
        i += 1
    return pts  

@dataclass
class FieldZones:
    
    li: Tuple[Tuple[float, float], Tuple[float, float]]
    ri: Tuple[Tuple[float, float], Tuple[float, float]]
    lo: Tuple[Tuple[float, float], Tuple[float, float]]
    ro: Tuple[Tuple[float, float], Tuple[float, float]]

def make_field_zones(point_list: List[Tuple[float, float]]) -> FieldZones:
    P = point_list  
    
    
    def box(a, b):
        (x1, y1), (x2, y2) = a, b
        return ((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)))

    
    li = box((P[4][0], P[8][1]), (0.0, P[4][1]))
    
    ri = box((0.0, P[11][1]), (P[7][0], P[7][1]))
    
    lo = box((min(P[2][0], P[7][0], P[10][0]),
              min(P[6][1], P[11][1], P[14][1])),
             (max(P[3][0], P[3][0], P[15][0]),
              max(P[2][1], P[7][1], P[10][1])))
    
    ro = box((min(P[0][0], P[0][0], P[12][0]),
              min(P[5][1], P[8][1], P[13][1])),
             (max(P[1][0], P[4][0], P[13][0]),
              max(P[0][1], P[4][1], P[8][1])))

    return FieldZones(li=li, ri=ri, lo=lo, ro=ro)

def in_box(x: float, y: float, box: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
    (xmin, ymin), (xmax, ymax) = box
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

class BallPlaceChecker:
    def __init__(self, zones: FieldZones):
        self.z = zones
        
        self.flags = {
            "On Floor": False,
            "Hitted": False,
            "Thrower": False,
            "OutLined": False,
            "L In": False,
            "R In": False,
            "L Out": False,
            "R Out": False,
            "Running": False,
        }

    def check(self, bx: float, by: float) -> Optional[str]:
        
        for k in self.flags.keys():
            self.flags[k] = False

        if in_box(bx, by, self.z.li):
            self.flags["L In"] = True
            return "li"
        if in_box(bx, by, self.z.ri):
            self.flags["R In"] = True
            return "ri"
        if in_box(bx, by, self.z.lo):
            self.flags["L Out"] = True
            return "lo"
        if in_box(bx, by, self.z.ro):
            self.flags["R Out"] = True
            return "ro"
        return None




stop_flag = False
streams: Dict[int, CamStream] = {}

def camera_thread(cam_id: int, device_id: int):
    global stop_flag
    ts = time.strftime('%X')
    print(f"[{ts}] [INFO] Starting camera thread cam:{cam_id} dev:{device_id}")

    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)  
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\033[31m[{time.strftime('%X')}] [ERROR] Failed to open cam:{cam_id} dev:{device_id}\033[0m")
        streams[cam_id] = CamStream(None, deque(maxlen=1))
        return

    streams[cam_id] = CamStream(cap, deque(maxlen=1))
    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            streams[cam_id].frames.append(frame)
        else:
            print(f"\033[31m[{time.strftime('%X')}] [ERROR] Failed to read frame cam:{cam_id}\033[0m")

    cap.release()
    print(f"[{time.strftime('%X')}] [INFO] Camera thread {cam_id} stopped.")




def main():
    global stop_flag,camera_cfgs

    camera_cfgs = {}
    for i in range(camera_count):
        w = int(input(f"Camera {i+1} Pixel Width:"))
        h = int(input(f"Camera {i+1} Pixel Height:"))
        alpha_h = int(input(f"Camera {i+1} Horizontal FOV:"))
        alpha_v = int(input(f"Camera {i+1} Vertical FOV:"))
        camera_cfgs[i] = [w,h,alpha_h,alpha_v]
    
    camera_config_input = []
    
    for cam_id, dev_id in camera_indices.items():
        t = Thread(target=camera_thread, args=(cam_id, dev_id), daemon=True)
        t.start()

    
    t0 = now()
    while len(streams) < len(camera_indices) and now() - t0 < 3.0:
        time.sleep(0.01)

    
    if len(camera_indices) < 2:
        print("\033[31m[ERROR] Triangulation requires at least 2 cameras.\033[0m")
        stop_flag = True
        return

    
    if sum(1 for s in streams.values() if s.cap is None) >= len(camera_indices):
        print("\033[31m[ERROR] All cameras failed to open.\033[0m")
        stop_flag = True
        return

    
    camera_configs = load_camera_configs()
    calibrate = CameraCalibration(camera_configs, 800, 800, FRAME_WIDTH, FRAME_HEIGHT)
    camera_params = calibrate.get_camera_params()
    calibrate.print_projection_matrices()

    tracker = BallTracker3D(camera_params)

    vl: Dict[float, Tuple[float, float, float]] = {}
    gl: Dict[float, Tuple[float, float, float]] = {}
    pl: Dict[float, Tuple[float, float, float]] = {}

    
    point_list = build_point_grid()
    zones = make_field_zones(point_list)
    place_checker = BallPlaceChecker(zones)

    
    animate = Animation(vl, gl, pl)
    interface = UserInterface()
    animate.main()
    interface.update(place_checker.flags)

    
    for cam_id in camera_indices.keys():
        cv2.namedWindow(f"CAM{cam_id}", cv2.WINDOW_NORMAL)

    
    frame_interval = 1.0 / TARGET_FPS
    try:
        while True:
            loop_start = now()

            
            snapshot: Dict[int, any] = {}
            for cam_id, stream in streams.items():
                if stream.frames:
                    snapshot[cam_id] = stream.frames[-1]

            
            pts_2d: List[Tuple[float, float]] = []
            cam_ids: List[int] = []

            for cam_id, frame in snapshot.items():
                pt = tracker.detect_ball(frame)
                if pt is not None:
                    pts_2d.append(pt)
                    cam_ids.append(int(cam_id))

            if len(pts_2d) >= 2:
                position_3d = tracker.triangulate_point(pts_2d, cam_ids)
                tstamp = now()
                state = tracker.update_state(position_3d, tstamp)
                
                vl[tstamp] = state["velocity"]
                gl[tstamp] = state["direction"]
                pl[tstamp] = state["position"]

                bx, by = state["position"][0], state["position"][1]
                zone = place_checker.check(bx, by)
                
                interface.update(place_checker.flags)

            
            for cam_id, frame in snapshot.items():  
                cv2.imshow(f"CAM{cam_id}", frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break

            
            elapsed = now() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    finally:
        stop_flag = True

        time.sleep(0.1)
        cv2.destroyAllWindows()
