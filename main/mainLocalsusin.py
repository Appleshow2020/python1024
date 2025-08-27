import cv2
import time
from threading import Thread
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, Optional, Tuple, List
import numpy as np
import os
import glob
import json
import re

from classes.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D as BallTracker3D
from classes.CameraCalibration import CameraCalibration
from classes.UserInterface import UserInterface
from classes.CameraPOCalc import CameraPOCalc

# =========================
#  설정
# =========================
# cam_id -> device_id 매핑 (삼각측량은 최소 2대 필요)
camera_count: int = int(input("Camera Count:"))
camera_indices: Dict[int, int] = {}
reali = 0
for i in range(10):  # 0~10번 장치 탐색
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[{time.strftime('%X')}] [INFO] Camera found at index {i}")
        cap.release()
        camera_indices[reali] = i
        reali+=1
        
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 60  # 메인 루프 처리 목표 fps

# =========================
#  유틸 · 데이터 구조
# =========================
@dataclass
class CamStream:
    cap: Optional[cv2.VideoCapture]
    frames: Deque  # latest-only queue

def now() -> float:
    # 단조 증가 타임스탬프(고정밀)
    return time.perf_counter()

def now2() -> str:
    return time.strftime("%X")

def build_point_grid() -> List[Tuple[float, float]]:
    # 원래 로직의 pdx/pdy 기반 4x4 격자 생성
    pdx = [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11]
    pdy = [7, 4, -4, -7]
    pts = []
    i = 0
    while len(pts) != 16:
        pts.append((pdx[i], pdy[i // 4]))
        i += 1
    return pts  # [(x, y), ...] length=16

@dataclass
class FieldZones:
    # 네 영역을 사전 계산된 사각형으로 정의: ((xmin, ymin), (xmax, ymax))
    li: Tuple[Tuple[float, float], Tuple[float, float]]
    ri: Tuple[Tuple[float, float], Tuple[float, float]]
    lo: Tuple[Tuple[float, float], Tuple[float, float]]
    ro: Tuple[Tuple[float, float], Tuple[float, float]]

def make_field_zones(point_list: List[Tuple[float, float]]) -> FieldZones:
    P = point_list  # 가독성
    # 기존 조건식을 해석하여 사각형 포함 테스트로 재정의
    # 좌표계/인덱스는 원 코드의 의미를 유지하되, xmin<=x<=xmax & ymin<=y<=ymax 형태로 정규화
    def box(a, b):
        (x1, y1), (x2, y2) = a, b
        return ((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)))

    # li: x in [P[4].x, 0], y in [P[8].y, P[4].y]
    li = box((P[4][0], P[8][1]), (0.0, P[4][1]))
    # ri: x in [0, P[7].x], y in [P[11].y, P[7].y]
    ri = box((0.0, P[11][1]), (P[7][0], P[7][1]))
    # lo: 세 구역을 합친 영역 → 우선순위로 lo를 하나의 큰 바운딩 박스로 단순화(원래보다 약간 관대하지만 배타 우선순위로 처리)
    lo = box((min(P[2][0], P[7][0], P[10][0]),
              min(P[6][1], P[11][1], P[14][1])),
             (max(P[3][0], P[3][0], P[15][0]),
              max(P[2][1], P[7][1], P[10][1])))
    # ro: 세 구역을 합친 영역
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
        # 상태 dict 사용
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
        # 배타적 우선순위: inside(L/R) → outside(L/R) 순으로 평가
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

# =========================
#  카메라 스레드
# =========================
stop_flag = False
streams: Dict[int, CamStream] = {}

def camera_thread(cam_id: int, device_id: int):
    global stop_flag
    ts = time.strftime('%X')
    print(f"[{ts}] [INFO] Starting camera thread cam:{cam_id} dev:{device_id}")

    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)  # 플랫폼에 맞게 변경 필요
    # 저지연 설정(장치가 지원하는 경우에만 유효)
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

def ask(prompt, cast):
    while True:
        try:
            return cast(input(prompt))
        except:
            print("retry.")

def list_presets():
    files = glob.glob("preset*.json")
    presets = {}
    for f in files:
        m = re.match(r"preset(\d+)\.json", os.path.basename(f))
        if m:
            presets[int(m.group(1))] = f
    return dict(sorted(presets.items()))

def load_preset():
    presets = list_presets()
    if not presets:
        print(f"\033[31m[{now2()}] [ERROR] No Preset to load\033[0m")
        return None

    print(f"[{now2()}] [INFO] List of preset:", ", ".join(map(str, presets.keys())))
    while True:
        try:
            num = int(input("Preset to check: "))
            if num not in presets:
                print(f"\033[31m[{now2()}] [ERROR] No Preset number {num}")
                continue
            with open(presets[num], "r", encoding="utf-8") as fp:
                data = json.load(fp)
            print("Loaded preset detail", data)
            yn = input("Use this preset? (y/n): ").lower()
            if yn == "y":
                return data
        except Exception as e:
            print("retry.", e)

def save_preset(data, num=None):
    if num is None:  # 번호 자동 할당
        existing = list_presets().keys()
        num = max(existing) + 1 if existing else 1
    with open(f"preset{num}.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved preset{num}.json.")
    return num

def get_camera_config(idx, use_preset=False):
    if use_preset:
        data = load_preset()
        if data: 
            return list(data.values())
        print(f"\033[31m[{now2()}] [ERROR] Failed to load preset.\033[0m")

    print(f"\n=== Camera {idx} Input ===")
    W = ask("Width Resolution (eg: 1920): ", int)
    H = ask("Height Resolution (eg: 1080): ", int)
    alpha_h = np.radians(ask("Horizontal FOV (deg): ", float))
    alpha_v = np.radians(ask("Vertical FOV (deg): ", float))
    pmin, pmax = ask("pitch bound (deg, eg: -90 0): ", lambda x: tuple(map(float, x.split())))
    rmin, rmax = ask("roll bound (deg, eg: 0 359): ", lambda x: tuple(map(float, x.split())))
    a = ask("Observation Area Length of Width: ", float)
    b = ask("Observation Area Length of Height: ", float)
    O = ask("Area Center O (x y z) 3 float: ", lambda x: tuple(map(float, x.split())))
    phi = np.radians(ask("Area Rotation phi (deg): ", float))

    P_list = [
        [O[0]+a/2, O[1]+b/2, 0],
        [O[0]-a/2, O[1]+b/2, 0],
        [O[0]-a/2, O[1]-b/2, 0],
        [O[0]+a/2, O[1]-b/2, 0],
    ]

    data = {
        "W": W, "H": H,
        "alpha_h": alpha_h, "alpha_v": alpha_v,
        "theta_p_bounds": (np.radians(pmin), np.radians(pmax)),
        "theta_r_bounds": (np.radians(rmin), np.radians(rmax)),
        "a": a, "b": b,
        "O": O, "phi": phi,
        "P_list": P_list
    }

    return data

def set_camera_config(camera_configs):
    result = []
    for idx,cfg in enumerate(camera_configs):
        cfg = cfg[idx]
        solver = CameraPOCalc(W=cfg["W"], H=cfg["H"],
                              alpha_h=cfg["alpha_h"], alpha_v=cfg["alpha_v"],
                              theta_p_bounds=cfg["theta_p_bounds"],
                              theta_r_bounds=cfg["theta_r_bounds"])
        temp = solver.solve(a=cfg["a"], b=cfg["b"],
                           O=cfg["O"], phi=cfg["phi"],
                           P_list=cfg["P_list"])
        if temp["success"] == False:
            print(f"\033[31m[{now2()}] [ERROR] Failed to calculate Camera{idx+1}. \033[0m")
        else:
            whattoappend={"id": f"cam{idx+1}", "position": list(temp["C"]), "rotation":[temp["pitch"], temp["yaw"], temp["roll"]]}
            result.append(whattoappend)
    print(*result,sep='\n')
    return result

# =========================
#  메인
# =========================
def main():
    global stop_flag

    # 카메라 스레드 시작
    for cam_id, dev_id in camera_indices.items():
        t = Thread(target=camera_thread, args=(cam_id, dev_id), daemon=True)
        t.start()

    # 충분한 카메라가 준비될 때까지 대기(타임아웃 포함)
    t0 = now()
    while len(streams) < len(camera_indices) and now() - t0 < 3.0:
        time.sleep(0.01)

    # 삼각측량 최소 카메라 수 체크
    if len(camera_indices) < 2:
        print(f"\033[31m[{now2()}] [ERROR] Triangulation requires at least 2 cameras.\033[0m")

    # VideoCapture 오픈 실패 카메라가 과반이면 종료
    if sum(1 for s in streams.values() if s.cap is None) >= len(camera_indices):
        print(f"\033[31m[{now2()}] [ERROR] All cameras failed to open.\033[0m")

    # 캘리브레이션/트래커 초기화
    use_preset = input("Load Preset? (y/n): ").lower() == "y"
    camera_configs = []
    if use_preset:
        config = get_camera_config(1, use_preset=True)
        for i in range(len(config)):
            camera_configs.append(config)
    else:
        camera_configs = [get_camera_config(i+1) for i in range(camera_count)]
        whattosave={}
        for idx,i in enumerate(camera_configs):
            whattosave[idx] = i
        save_preset(whattosave)
    new_camera_configs = set_camera_config(camera_configs)
    calibrate = CameraCalibration(new_camera_configs, 800, 800, FRAME_WIDTH, FRAME_HEIGHT)
    camera_params = calibrate.get_camera_params()
    calibrate.print_projection_matrices()

    tracker = BallTracker3D(camera_params)

    vl: Dict[float, Tuple[float, float, float]] = {}
    gl: Dict[float, Tuple[float, float, float]] = {}
    pl: Dict[float, Tuple[float, float, float]] = {}

    # 필드 영역 준비
    point_list = build_point_grid()
    zones = make_field_zones(point_list)
    place_checker = BallPlaceChecker(zones)

    # 애니메이션/UI
    animate = Animation(vl, gl, pl)
    interface = UserInterface()
    animate.main()
    interface.update(place_checker.flags)

    # 창은 미리 생성
    for cam_id in camera_indices.keys():
        cv2.namedWindow(f"CAM{cam_id}", cv2.WINDOW_NORMAL)

    frame_interval = 1.0 / TARGET_FPS
    try:
        while True:
            loop_start = now()

            # 프레임 스냅샷(레이스 회피: 참조만 복사)
            snapshot: Dict[int, any] = {}
            for cam_id, stream in streams.items():
                if stream.frames:
                    snapshot[cam_id] = stream.frames[-1]

            # 포인트 수집
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
                # print(state)  # 필요시 유지
                vl[tstamp] = state["velocity"]
                gl[tstamp] = state["direction"]
                pl[tstamp] = state["position"]

                bx, by = state["position"][0], state["position"][1]
                zone = place_checker.check(bx, by)
                # UI에 최신 플래그 반영
                interface.update(place_checker.flags)

            # 디스플레이 업데이트
            for cam_id, frame in snapshot.items():  
                cv2.imshow(f"CAM{cam_id}", frame)

            # 키 이벤트/루프 페이싱
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break

            # 일정한 루프 주기 유지(불필요한 busy-wait 제거)
            elapsed = now() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    finally:
        stop_flag = True
        # 스레드는 데몬이므로 종료 시점에 자동 소멸되지만, 잠시 대기
        time.sleep(0.1)
        cv2.destroyAllWindows()
main()