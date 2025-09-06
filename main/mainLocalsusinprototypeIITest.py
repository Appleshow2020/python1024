import cv2
import time
from threading import Thread, Event
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, Optional, Tuple, List
import numpy as np
import os
import glob
import json
import re
import queue
import matplotlib
matplotlib.use('TkAgg')  # GUI 백엔드 설정
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from classes.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D as BallTracker3D
from classes.CameraCalibration import CameraCalibration
from classes.UserInterface import UserInterface
from classes.CameraPOCalc import CameraPOCalc
from classes.printing import *


# 전역 설정
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TARGET_FPS = 60

CAM1POS = [-0.47,-0.52,0.19]; CAM1ROT = [-30,45,-10]
CAM2POS = [0.05,0.05,0.62]; CAM2ROT = [-90,0,100]
CAM3POS = [0.61,0.39,0.19]; CAM3ROT = [-20,-120,0]

@dataclass
class CamStream:
    cap: Optional[cv2.VideoCapture]
    frames: Deque
    last_detection: Optional[Tuple[float, float]] = None

@dataclass
class FieldZones:
    li: Tuple[Tuple[float, float], Tuple[float, float]]
    ri: Tuple[Tuple[float, float], Tuple[float, float]]
    lo: Tuple[Tuple[float, float], Tuple[float, float]]
    ro: Tuple[Tuple[float, float], Tuple[float, float]]

class AnimationWrapper:
    """Animation 클래스의 래퍼 - 메인 스레드에서 matplotlib 실행"""
    def __init__(self, original_animation):
        self.animation = original_animation
        self.last_update = 0
        self.data_queue = queue.Queue()
        self.animation_thread = None
        self.stop_event = Event()
        self.ui_update_event = Event()
        
        # matplotlib 초기화를 메인 스레드에서 수행
        self.fig = None
        self.ax = None
        self.init_matplotlib()
        
    def init_matplotlib(self):
        """matplotlib 초기화 (메인 스레드에서 호출)"""
        try:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)
            self.ax.set_zlim(0, 20)
            self.ax.set_title("3D Ball Tracking")
            plt.ion()  # 인터랙티브 모드 활성화
            printf("Matplotlib initialized in main thread", LT.info)
        except Exception as e:
            printf(f"Matplotlib initialization failed: {e}", LT.error)
            
    def update_data(self, vl=None, gl=None, pl=None):
        """데이터 업데이트 (스레드 안전)"""
        current_time = now()
        if current_time - self.last_update > 0.5:  # 0.5초마다 업데이트
            self.last_update = current_time
            try:
                # 큐에 데이터 추가
                data = {
                    'vl': vl or {},
                    'gl': gl or {},
                    'pl': pl or {},
                    'timestamp': current_time
                }
                if not self.data_queue.full():
                    self.data_queue.put(data)
                    self.ui_update_event.set()
                printf("Animation data queued for update", LT.info)
                return True
            except Exception as e:
                printf(f"Animation data update failed: {e}", LT.warning)
        return False
    
    def process_updates(self):
        """메인 스레드에서 호출할 업데이트 처리"""
        if self.ui_update_event.is_set():
            try:
                while not self.data_queue.empty():
                    data = self.data_queue.get_nowait()
                    self._update_plot(data)
                self.ui_update_event.clear()
            except queue.Empty:
                pass
            except Exception as e:
                printf(f"Process updates failed: {e}", LT.warning)
    
    def _update_plot(self, data):
        """플롯 업데이트 (메인 스레드에서만 호출)"""
        if self.fig is None or self.ax is None:
            return
            
        try:
            self.ax.clear()
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)
            self.ax.set_title("3D Ball Tracking - Live")
            
            # 위치 데이터 플롯
            pl = data.get('pl', {})
            if pl:
                positions = list(pl.values())
                if positions:
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions]
                    self.ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.7, label='Ball Position')
                    
                    # 궤적 그리기
                    if len(positions) > 1:
                        self.ax.plot(x_coords, y_coords, 'r-', alpha=0.5, linewidth=2)
            
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            plt.draw()
            plt.pause(0.001)  # 짧은 pause로 화면 업데이트
            
        except Exception as e:
            printf(f"Plot update failed: {e}", LT.warning)
    
    def main(self):
        """메인 애니메이션 호출 (메인 스레드에서만)"""
        try:
            if hasattr(self.animation, 'main'):
                self.animation.main()
            else:
                # 기본 애니메이션 처리
                self.process_updates()
        except Exception as e:
            printf(f"Animation main failed: {e}", LT.warning)
    
    def close(self):
        """정리 함수"""
        self.stop_event.set()
        if self.fig is not None:
            plt.close(self.fig)

class UIWrapper:
    """UserInterface 래퍼 - 더 안정적인 업데이트"""
    def __init__(self, original_ui):
        self.ui = original_ui
        self.last_update = 0
        self.last_flags = {}
        
    def update(self, flags):
        """UI 업데이트 (변경사항이 있을 때만)"""
        current_time = now()
        
        # 플래그가 변경되었거나 1초가 지났을 때만 업데이트
        if (flags != self.last_flags) or (current_time - self.last_update > 1.0):
            try:
                if hasattr(self.ui, 'update'):
                    self.ui.update(flags)
                self.last_flags = flags.copy()
                self.last_update = current_time
                printf(f"UI updated: {flags}", LT.info)
                return True
            except Exception as e:
                printf(f"UI update failed: {e}", LT.warning)
        return False

class BallPlaceChecker:
    """볼 위치 체커"""
    def __init__(self, zones: FieldZones):
        self.z = zones
        self.flags = {
            "On Floor": False, "Hitted": False, "Thrower": False, "OutLined": False,
            "L In": False, "R In": False, "L Out": False, "R Out": False, "Running": False,
        }

    def check(self, bx: float, by: float) -> Optional[str]:
        """볼 위치 체크"""
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

def now() -> float:
    return time.perf_counter()

def find_and_select_cameras():
    camera_count: int = int(input("Camera Count: "))
    available_cameras: Dict[int, int] = {}
    selected_cameras: Dict[int, int] = {}
    
    printf("Searching for the camera...", LT.info)
    # 0~10번 장치 탐색
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                printf(f"Camera found at index {i}", LT.info)
                cap.release()
                available_cameras[len(available_cameras)] = i
            else:
                cap.release()
        except Exception as e:
            printf(f"Error checking camera {i}: {e}", LT.warning)
    
    printf(f"Found {len(available_cameras)} camera in total.", LT.info)
    
    if len(available_cameras) == 0:
        printf("No available cameras found!", LT.error)
        return {}
    
    # 카메라 선택
    for _, device_idx in available_cameras.items():
        if len(selected_cameras) >= camera_count:
            break
            
        try:
            cap = cv2.VideoCapture(device_idx)
            if not cap.isOpened():
                continue
                
            printf(f"Testing Camera Device {device_idx}", LT.info)
            print("Options:")
            print("  't' - Accept this camera")
            print("  'p' - Pass (skip this camera)")
            print("  '0-9' - Assign specific camera number (0-9)")
            print("  'q' - Quit selection")
            
            frame_shown = False
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_shown = True
                    # 화면에 현재 상태 표시
                    info_text = f"Device {device_idx} | Selected: {len(selected_cameras)}/{camera_count}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    options_text = "t:Accept, p:Pass, 0-9:Set cam#, q:Quit"
                    cv2.putText(frame, options_text, (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow(f"Camera Selection - Device {device_idx}", frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('t'):
                        # 자동으로 다음 카메라 번호 할당
                        cam_id = len(selected_cameras)
                        selected_cameras[cam_id] = device_idx
                        printf(f"Camera {cam_id} -> Device {device_idx} selected (auto-assigned).", LT.success)
                        break
                        
                    elif key == ord('p'):
                        printf(f"Device {device_idx} passed (skipped).", LT.info)
                        break
                        
                    elif key >= ord('0') and key <= ord('9'):
                        # 특정 카메라 번호 할당
                        cam_id = int(chr(key))
                        if cam_id in selected_cameras:
                            printf(f"Camera {cam_id} is already assigned! Choose different number.", LT.warning)
                            continue
                        selected_cameras[cam_id] = device_idx
                        printf(f"Camera {cam_id} -> Device {device_idx} selected (manual assignment).", LT.success)
                        break
                        
                    elif key == ord('q'):
                        printf("Selection terminated by user.", LT.info)
                        cap.release()
                        cv2.destroyAllWindows()
                        return selected_cameras
                        
                else:
                    if not frame_shown:
                        printf("Failed to grab frame from camera", LT.warning)
                        break
                    time.sleep(0.01)
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            printf(f"Error processing Camera Device {device_idx}: {e}", LT.error)
    
    printf(f"Camera selection completed. Selected {len(selected_cameras)} cameras.", LT.info)
    for cam_id, dev_id in selected_cameras.items():
        printf(f"  Camera {cam_id} -> Device {dev_id}", LT.info)
    
    return selected_cameras

def build_point_grid() -> List[Tuple[float, float]]:
    """4x4 격자 생성"""
    pdx = [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11]
    pdy = [7, 4, -4, -7]
    pts = []
    i = 0
    while len(pts) != 16:
        pts.append((pdx[i], pdy[i // 4]))
        i += 1
    return pts

def make_field_zones(point_list: List[Tuple[float, float]]) -> FieldZones:
    """필드 영역 정의"""
    P = point_list
    def box(a, b):
        (x1, y1), (x2, y2) = a, b
        return ((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)))

    li = box((P[4][0], P[8][1]), (0.0, P[4][1]))
    ri = box((0.0, P[11][1]), (P[7][0], P[7][1]))
    lo = box((min(P[2][0], P[7][0], P[10][0]), min(P[6][1], P[11][1], P[14][1])),
             (max(P[3][0], P[3][0], P[15][0]), max(P[2][1], P[7][1], P[10][1])))
    ro = box((min(P[0][0], P[0][0], P[12][0]), min(P[5][1], P[8][1], P[13][1])),
             (max(P[1][0], P[4][0], P[13][0]), max(P[0][1], P[4][1], P[8][1])))
    
    return FieldZones(li=li, ri=ri, lo=lo, ro=ro)

def in_box(x: float, y: float, box: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
    """점이 박스 안에 있는지 확인"""
    (xmin, ymin), (xmax, ymax) = box
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

stop_flag = False
streams: Dict[int, CamStream] = {}

def camera_thread(cam_id: int, device_id: int):
    global stop_flag
    printf(f"Starting camera thread cam:{cam_id} dev:{device_id}", LT.info)

    try:
        cap = cv2.VideoCapture(device_id)
        
        # 카메라 설정 시도
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            printf(f"Camera {cam_id} configuration failed: {e}", LT.warning)

        if not cap.isOpened():
            printf(f"Failed to open cam:{cam_id} dev:{device_id}", LT.error)
            streams[cam_id] = CamStream(None, deque(maxlen=5))
            return

        streams[cam_id] = CamStream(cap, deque(maxlen=5))
        frame_count = 0
        consecutive_failures = 0
        
        while not stop_flag:
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 프레임 크기 확인
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        streams[cam_id].frames.append(frame.copy())
                        frame_count += 1
                        consecutive_failures = 0
                        
                        if frame_count % 120 == 0:  # 2초마다 로그
                            printf(f"Camera {cam_id}: {frame_count} frames captured", LT.debug)
                    else:
                        printf(f"Camera {cam_id}: Invalid frame size", LT.warning)
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        printf(f"Camera {cam_id}: Too many consecutive failures", LT.error)
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                printf(f"Camera {cam_id} read error: {e}", LT.error)
                time.sleep(0.1)

    except Exception as e:
        printf(f"Camera thread {cam_id} initialization failed: {e}", LT.error)
        streams[cam_id] = CamStream(None, deque(maxlen=5))
    finally:
        if 'cap' in locals():
            cap.release()
        printf(f"Camera thread {cam_id} stopped.", LT.info)

def debug_ball_detection(frame, cam_id):
    """볼 검출 과정을 시각화하여 디버깅"""
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), False
    
    try:
        # 다양한 색상 범위로 테스트
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 기존 범위
        lower_color1 = (0, 36, 2)
        upper_color1 = (179, 216, 182)
        
        # 더 넓은 범위들
        lower_color2 = (0, 50, 50)
        upper_color2 = (15, 255, 255)
        
        lower_color3 = (160, 50, 50)
        upper_color3 = (179, 255, 255)
        
        # 여러 마스크 결합
        mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
        mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
        mask3 = cv2.inRange(hsv, lower_color3, upper_color3)
        
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
        
        # 형태학적 연산
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        result = frame.copy()
        result[combined_mask > 0] = (0, 255, 0)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_detection = False
        if contours:
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > 50:  # 더 낮은 임계값
                    has_detection = True
                    cv2.drawContours(result, [c], -1, (255, 0, 0), 2)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(result, f"Area: {int(area)}", (cx-30, cy-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 디버그 정보
        info_text = f"CAM{cam_id} - Contours: {len(contours)}, Detection: {'YES' if has_detection else 'NO'}"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result, has_detection
        
    except Exception as e:
        printf(f"Debug detection failed for cam {cam_id}: {e}", LT.error)
        return frame.copy(), False

def main():
    global stop_flag

    printf("Ball Tracking System Starting", LT.info)
    
    # 1. 카메라 설정
    camera_indices = find_and_select_cameras()
    
    if len(camera_indices) == 0:
        printf("No cameras found.", LT.error)
        return
        
    if len(camera_indices) < 2:
        printf(f"At least 2 cameras are recommended for triangulation. Current: {len(camera_indices)}", LT.warning)

    printf(f"Final selected cameras: {camera_indices}", LT.info)

    # 2. 카메라 스레드 시작
    for cam_id, dev_id in camera_indices.items():
        t = Thread(target=camera_thread, args=(cam_id, dev_id), daemon=True)
        t.start()

    # 3. 카메라 준비 대기
    printf("Initializing cameras...", LT.info)
    t0 = now()
    while len(streams) < len(camera_indices) and now() - t0 < 10.0:
        time.sleep(0.2)
        active_cams = sum(1 for s in streams.values() if s.cap is not None)
        printf(f"Active cameras: {active_cams}/{len(camera_indices)}", LT.debug)

    active_cameras = sum(1 for s in streams.values() if s.cap is not None)
    printf(f"Final active cameras: {active_cameras}", LT.info)

    # 4. 캘리브레이션
    printf("Setting up calibration...", LT.info)
    try:
        calibrate = CameraCalibration([
            {"id": "cam1", "position": CAM1POS, "rotation":CAM1ROT},
            {"id": "cam2", "position": CAM2POS, "rotation":CAM2ROT},
            {"id": "cam3", "position": CAM3POS, "rotation":CAM3ROT}
        ], FRAME_WIDTH, FRAME_HEIGHT, 800, 800)
        
        camera_params = calibrate.get_camera_params()
        calibrate.print_projection_matrices()
    except Exception as e:
        printf(f"aled: {e}", LT.error)
        return

    # 5. 트래커 및 시각화 초기화
    tracker = BallTracker3D(camera_params)
    
    vl: Dict[float, Tuple[float, float, float]] = {}
    gl: Dict[float, Tuple[float, float, float]] = {}
    pl: Dict[float, Tuple[float, float, float]] = {}
    
    point_list = build_point_grid()
    zones = make_field_zones(point_list)
    place_checker = BallPlaceChecker(zones)
    
    # 애니메이션 및 UI 초기화 (메인 스레드에서)
    animate = None
    interface = None
    try:
        original_animation = Animation(vl, gl, pl)
        animate = AnimationWrapper(original_animation)
        
        try:
            original_ui = UserInterface()
            interface = UIWrapper(original_ui)
            interface.update(place_checker.flags)
            printf("Animation and Interface initialized", LT.info)
        except Exception as e:
            printf(f"UI initialization failed: {e}", LT.warning)
            interface = None
        
    except Exception as e:
        printf(f"Failed to initialize Animation: {e}", LT.warning)
        animate = None

    # 6. 창 생성
    printf("Creating windows...", LT.info)
    try:
        for cam_id in camera_indices.keys():
            cv2.namedWindow(f"CAM{cam_id}", cv2.WINDOW_NORMAL)
            cv2.namedWindow(f"DEBUG{cam_id}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"CAM{cam_id}", 640, 360)
            cv2.resizeWindow(f"DEBUG{cam_id}", 640, 360)
    except Exception as e:
        printf(f"Window creation error: {e}", LT.warning)

    # 메인 루프
    frame_interval = 1.0 / 30  # 30 FPS로 낮춤
    detection_count = 0
    triangulation_count = 0
    loop_count = 0
    
    printf("Main loop starting", LT.info)
    printf("'q': quit, 'd': toggle debug, 'a': update animation, 'p': show plot", LT.info)
    
    debug_mode = True
    
    try:
        while True:
            loop_start = now()
            loop_count += 1

            # 프레임 수집
            snapshot: Dict[int, any] = {}
            for cam_id, stream in streams.items():
                if stream.cap is not None and stream.frames and len(stream.frames) > 0:
                    snapshot[cam_id] = stream.frames[-1]

            if not snapshot:
                printf("No available frames", LT.debug)
                time.sleep(0.1)
                continue

            # 볼 검출
            pts_2d: List[Tuple[float, float]] = []
            cam_ids: List[int] = []
            
            for cam_id, frame in snapshot.items():
                if frame is None:
                    continue
                    
                try:
                    if debug_mode:
                        debug_frame, has_detection = debug_ball_detection(frame, cam_id)
                        cv2.imshow(f"DEBUG{cam_id}", debug_frame)
                    
                    # 볼 검출 시도
                    pt = tracker.detect_ball(frame)
                    if pt is not None:
                        pts_2d.append(pt)
                        cam_ids.append(cam_id)
                        detection_count += 1
                        streams[cam_id].last_detection = pt
                        
                        # 검출 표시
                        cv2.circle(frame, tuple(map(int, pt)), 10, (0, 255, 0), 2)
                        cv2.putText(frame, f"Ball: {pt}", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # 상태 정보 표시
                    status_text = f"Det: {detection_count}, Tri: {triangulation_count}, Loop: {loop_count}"
                    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imshow(f"CAM{cam_id}", frame)
                    
                except Exception as e:
                    printf(f"Frame processing error cam{cam_id}: {e}", LT.error)

            # 삼각측량 시도
            if len(pts_2d) >= 2:
                if loop_count % 30 == 0:  # 30번에 한 번만 로그
                    printf(f"Attempting triangulation with {len(pts_2d)} cameras", LT.debug)
                
                try:
                    position_3d = tracker.triangulate_point(pts_2d, cam_ids)
                    
                    if not np.any(np.isnan(position_3d)):
                        triangulation_count += 1
                        timestamp = now()
                        state = tracker.update_state(position_3d, timestamp)
                        
                        # 데이터 저장
                        if state['velocity'] is not None and not any(np.isnan(state['velocity'])):
                            vl[timestamp] = tuple(state['velocity'])
                        if state['direction'] is not None and not any(np.isnan(state['direction'])):
                            gl[timestamp] = tuple(state['direction'])
                        if state['position'] is not None and not any(np.isnan(state['position'])):
                            pl[timestamp] = tuple(state['position'])
                        
                        # 위치 체크 및 UI 업데이트
                        if place_checker is not None:
                            bx, by = state["position"][0], state["position"][1]
                            zone = place_checker.check(bx, by)
                        
                        if interface is not None:
                            try:
                                interface.update(place_checker.flags)
                            except Exception as e:
                                if loop_count % 100 == 0:  # 가끔만 로그
                                    printf(f"UI update failed: {e}", LT.warning)
                        
                        # 애니메이션 데이터 업데이트
                        if animate is not None:
                            animate.update_data(vl, gl, pl)
                        
                        if triangulation_count % 10 == 0:  # 10번에 한 번만 로그
                            printf(f"Triangulation #{triangulation_count}: {position_3d[:2]}", LT.success)
                        
                except Exception as e:
                    if loop_count % 50 == 0:  # 50번에 한 번만 로그
                        printf(f"Triangulation failed: {e}", LT.error)

            # 메인 스레드에서 애니메이션 및 UI 업데이트 처리
            if animate is not None:
                animate.process_updates()

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_flag = True
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                printf(f"Debug mode: {'ON' if debug_mode else 'OFF'}", LT.info)
            elif key == ord('a') and animate is not None:
                try:
                    animate.main()
                except Exception as e:
                    printf(f"Manual animation failed: {e}", LT.warning)
            elif key == ord('p') and animate is not None:
                try:
                    # 현재 데이터로 즉시 플롯 업데이트
                    data = {'vl': vl, 'gl': gl, 'pl': pl, 'timestamp': now()}
                    animate._update_plot(data)
                except Exception as e:
                    printf(f"Manual plot update failed: {e}", LT.warning)

            # FPS 제한
            elapsed = now() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        printf("Terminating due to keyboard interrupt", LT.info)
    except Exception as e:
        printf(f"Main loop error: {e}", LT.error)
        import traceback
        traceback.print_exc()
    finally:
        printf("Cleaning up...", LT.info)
        stop_flag = True
        time.sleep(0.3)
        
        # 애니메이션 정리
        if animate is not None:
            animate.close()
        
        cv2.destroyAllWindows()
        
        printf("=== Execution Statistics ===", LT.info)
        printf(f"Total loops: {loop_count}", LT.info)
        printf(f"Total detections: {detection_count}", LT.info)
        printf(f"Total triangulations: {triangulation_count}", LT.info)

if __name__ == "__main__":
    main()