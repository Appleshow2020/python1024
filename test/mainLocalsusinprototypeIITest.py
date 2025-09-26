import cv2
import time
import cProfile
import pstats
from threading import Thread, Event, Lock
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Optional, Tuple, List, Set, Any
import numpy as np
import queue
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import logging
import logging.handlers
from pathlib import Path
import json  # JSON을 기본으로 사용

from classes.Animation import Animation
from classes.BallTracker3Dcopy import BallTracker3D as BallTracker3D
from classes.CameraCalibration import CameraCalibration
from classes.UserInterface import UserInterface
from classes.CameraPOCalc import CameraPOCalc
from classes.printing import *

# YAML 모듈 선택적 import
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    printf("PyYAML not installed. Using JSON config instead.", ptype=LT.warning)

# 설정 파일 로드
def load_config():
    """설정 파일 로드 또는 기본값 생성 - JSON 우선, YAML 선택적"""
    default_config = {
        'camera': {
            'width': 640,
            'height': 360,
            'fps': 30,
            'detection_interval': 3,
            'buffer_size': 3,
            'search_range': 10
        },
        'processing': {
            'update_intervals': {
                'ui': 1.0,
                'animation': 0.5,
                'stats': 5.0
            },
            'position_history_size': 100,
            'queue_size': 10
        },
        'detection': {
            'hsv_lower': [0, 50, 50],
            'hsv_upper': [15, 255, 255],
            'min_contour_area': 30,
            'morphology_kernel_size': 3,
            'enable_gpu': False,
            'model_path': 'ball_detection.onnx'
        },
        'display': {
            'plot_size': [8, 6],
            'max_plot_points': 10,
            'circle_radius': 5,
            'line_thickness': 2
        },
        'cameras': {
            '1': {"position": [-0.47, -0.52, 0.19], "rotation": [-30, 45, -10]},
            '2': {"position": [0.05, 0.05, 0.62], "rotation": [-90, 0, 100]},
            '3': {"position": [0.61, 0.39, 0.19], "rotation": [-20, -120, 0]}
        },
        'logging': {
            'level': 'INFO',
            'file': 'ball_tracker.log',
            'max_file_size': '10MB'
        },
        'profiling': {
            'enabled': False,
            'save_interval': 100,
            'output_dir': 'profiles'
        }
    }
    
    # JSON 파일 우선 확인
    json_config_path = Path("config.json")
    yaml_config_path = Path("config.yaml")
    
    config_path = None
    use_yaml = False
    
    if json_config_path.exists():
        config_path = json_config_path
        use_yaml = False
    elif yaml_config_path.exists() and YAML_AVAILABLE:
        config_path = yaml_config_path
        use_yaml = True
    else:
        # 새 파일 생성
        if YAML_AVAILABLE:
            config_path = yaml_config_path
            use_yaml = True
        else:
            config_path = json_config_path
            use_yaml = False
    
    # 기존 파일 로드
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                if use_yaml:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            printf(f"Configuration loaded from {config_path.name}", ptype=LT.info)
            return {**default_config, **config}
        except Exception as e:
            printf(f"Failed to load config: {e}. Using defaults.", ptype=LT.warning)
    
    # 새 설정 파일 생성
    try:
        with open(config_path, 'w') as f:
            if use_yaml:
                yaml.dump(default_config, f, default_flow_style=False)
            else:
                json.dump(default_config, f, indent=2)
        printf(f"Default {config_path.name} created", ptype=LT.info)
    except Exception as e:
        printf(f"Failed to create config file: {e}", ptype=LT.warning)
    
    return default_config

# 전역 설정 로드
CONFIG = load_config()

# 설정값 추출
FRAME_WIDTH = CONFIG['camera']['width']
FRAME_HEIGHT = CONFIG['camera']['height']
TARGET_FPS = CONFIG['camera']['fps']
DETECTION_INTERVAL = CONFIG['camera']['detection_interval']
BUFFER_SIZE = CONFIG['camera']['buffer_size']

class ProfiledBallDetector:
    """프로파일링이 포함된 볼 검출기"""
    def __init__(self):
        self.detection_times = deque(maxlen=100)
        self.gpu_available = False
        self.net = None
        self.profiler = None
        
        # GPU 가속 설정 시도
        if CONFIG['detection']['enable_gpu']:
            self._setup_gpu_detection()
        
        # OpenCV DNN 백엔드 최적화
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            printf(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}", ptype=LT.info)
        
        # 프로파일링 설정 (더 쉬운 방법)
        # 방법 1: 환경변수 사용
        self.enable_profiling = os.getenv('PROFILE', 'False').lower() == 'true'
        
        # 방법 2: 설정 파일에서 읽기
        if not self.enable_profiling:
            self.enable_profiling = CONFIG.get('profiling', {}).get('enabled', False)
        
        # 방법 3: 사용자 입력으로 결정
        if not self.enable_profiling:
            try:
                user_input = input("Enable profiling? (y/N): ").lower()
                self.enable_profiling = user_input == 'y'
            except:
                self.enable_profiling = False
        
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            printf("Profiling enabled", ptype=LT.info)
    
    def _setup_gpu_detection(self):
        """GPU 가속 볼 검출 설정"""
        model_path = CONFIG['detection']['model_path']
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNet(model_path)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.gpu_available = True
                    printf("GPU acceleration enabled for ball detection", ptype=LT.info)
                else:
                    printf("CUDA not available, using CPU", ptype=LT.warning)
            except Exception as e:
                printf(f"Failed to setup GPU detection: {e}", ptype=LT.warning)
    
    def detect_with_dnn(self, frame):
        """DNN을 사용한 볼 검출"""
        if self.net is None:
            return None, False
            
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            # 결과 처리 (YOLO 형식 가정)
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5 and class_id == 0:  # ball class
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        return (center_x, center_y), True
            
            return None, False
        except Exception as e:
            printf(f"DNN detection failed: {e}", ptype=LT.error)
            return None, False
    
    def detect_traditional(self, frame, cam_id, frame_count):
        """전통적인 컴퓨터 비전을 사용한 볼 검출"""
        if frame is None or frame.size == 0:
            return None, False
        
        # 프레임 스킵
        if frame_count % DETECTION_INTERVAL != 0:
            return None, False
        
        start_time = time.perf_counter()
        
        try:
            # GPU 메모리 사용 가능한 경우 GPU Mat 사용
            if self.gpu_available and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # GPU에서 크기 조정
                gpu_small = cv2.cuda.resize(gpu_frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
                
                # CPU로 다운로드
                small_frame = gpu_small.download()
            else:
                # CPU에서 처리
                small_frame = cv2.resize(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
            
            # HSV 변환
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # 설정된 색상 범위로 마스크 생성
            lower_color = np.array(CONFIG['detection']['hsv_lower'])
            upper_color = np.array(CONFIG['detection']['hsv_upper'])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # 형태학적 연산
            kernel_size = CONFIG['detection']['morphology_kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > CONFIG['detection']['min_contour_area']:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        # 원래 크기로 좌표 변환
                        cx = int(M["m10"] / M["m00"]) * 2
                        cy = int(M["m01"] / M["m00"]) * 2
                        
                        # 검출 시간 기록
                        detection_time = time.perf_counter() - start_time
                        self.detection_times.append(detection_time)
                        
                        return (cx, cy), True
            
            return None, False
            
        except Exception as e:
            printf(f"Traditional detection error cam{cam_id}: {e}", ptype=LT.error)
            return None, False
    
    def detect(self, frame, cam_id, frame_count):
        """통합 볼 검출 함수"""
        if self.enable_profiling and self.profiler:
            self.profiler.enable()
        
        # DNN 모델이 있으면 우선 사용, 없으면 전통적 방법 사용
        if self.net is not None:
            result = self.detect_with_dnn(frame)
        else:
            result = self.detect_traditional(frame, cam_id, frame_count)
        
        if self.enable_profiling and self.profiler:
            self.profiler.disable()
        
        return result
    
    def get_stats(self):
        """검출 성능 통계"""
        if not self.detection_times:
            return {}
        
        times = list(self.detection_times)
        return {
            'avg_detection_time': np.mean(times),
            'max_detection_time': np.max(times),
            'min_detection_time': np.min(times),
            'detection_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def save_profile(self, filename="detection_profile.prof"):
        """프로파일 결과 저장"""
        if self.enable_profiling and self.profiler:
            self.profiler.dump_stats(filename)
            
            # 텍스트 리포트도 생성
            with open(filename.replace('.prof', '.txt'), 'w') as f:
                stats = pstats.Stats(self.profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(file=f)
            
            printf(f"Profile saved to {filename}", ptype=LT.info)

@dataclass
class CamStream:
    """최적화된 카메라 스트림"""
    cap: Optional[cv2.VideoCapture]
    frames: Deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    last_detection: Optional[Tuple[float, float]] = None
    last_frame_time: float = 0.0
    detection_cache: Optional[Tuple[float, float]] = None
    cache_time: float = 0.0
    consecutive_failures: int = 0
    lock: Lock = field(default_factory=Lock)
    stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FieldZones:
    """필드 영역 정의"""
    li: Tuple[Tuple[float, float], Tuple[float, float]]
    ri: Tuple[Tuple[float, float], Tuple[float, float]]
    lo: Tuple[Tuple[float, float], Tuple[float, float]]
    ro: Tuple[Tuple[float, float], Tuple[float, float]]

class AdvancedAnimationWrapper:
    """고급 애니메이션 래퍼 - GPU 가속 및 최적화 포함"""
    def __init__(self):
        self.last_update = 0
        self.data_queue = queue.Queue(maxsize=CONFIG['processing']['queue_size'])
        self.stop_event = Event()
        self.ui_update_event = Event()
        self.fig = None
        self.ax = None
        self.plot_data = {'x': [], 'y': [], 'timestamps': []}
        self._init_matplotlib()
        
        # 성능 모니터링
        self.render_times = deque(maxlen=100)
        
    def _init_matplotlib(self):
        """matplotlib 초기화 및 최적화"""
        try:
            plt.ion()
            figsize = CONFIG['display']['plot_size']
            self.fig, self.ax = plt.subplots(figsize=figsize)
            
            # 블리팅 활성화로 성능 향상
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)
            self.ax.set_title("Advanced Ball Tracking")
            self.ax.grid(True, alpha=0.3)
            
            printf("Advanced matplotlib initialized", ptype=LT.info)
        except Exception as e:
            printf(f"Matplotlib init failed: {e}", ptype=LT.error)
            
    def update_data(self, pl: Dict):
        """최적화된 데이터 업데이트"""
        current_time = time.perf_counter()
        update_interval = CONFIG['processing']['update_intervals']['animation']
        
        if current_time - self.last_update < update_interval:
            return False
            
        try:
            # 큐 관리 - 최신 데이터만 유지
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self.data_queue.put({
                'pl': pl.copy(), 
                'timestamp': current_time,
                'data_points': len(pl)
            })
            self.ui_update_event.set()
            self.last_update = current_time
            return True
        except Exception as e:
            printf(f"Animation update failed: {e}", ptype=LT.warning)
        return False
    
    def process_updates(self):
        """고성능 업데이트 처리"""
        if not self.ui_update_event.is_set():
            return
            
        start_time = time.perf_counter()
        
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self._update_plot_advanced(data)
            self.ui_update_event.clear()
            
            # 렌더링 시간 기록
            render_time = time.perf_counter() - start_time
            self.render_times.append(render_time)
            
        except queue.Empty:
            pass
        except Exception as e:
            printf(f"Process updates failed: {e}", ptype=LT.warning)
    
    def _update_plot_advanced(self, data):
        """고급 플롯 업데이트 - 블리팅 사용"""
        if self.fig is None or self.ax is None:
            return
            
        try:
            pl = data.get('pl', {})
            if not pl:
                return
                
            max_points = CONFIG['display']['max_plot_points']
            positions = list(pl.values())[-max_points:]
            timestamps = list(pl.keys())[-max_points:]
            
            if not positions:
                return
                
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # 배경 복원 (블리팅)
            self.fig.canvas.restore_region(self.background)
            
            # 새로운 데이터 그리기
            if len(positions) > 0:
                # 점 그리기
                points = self.ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7, animated=True)
                
                # 궤적 그리기
                if len(positions) > 1:
                    line, = self.ax.plot(x_coords, y_coords, 'r-', alpha=0.5, linewidth=1, animated=True)
                    self.ax.draw_artist(line)
                
                self.ax.draw_artist(points)
            
            # 블리팅으로 업데이트
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
            
        except Exception as e:
            printf(f"Advanced plot update failed: {e}", ptype=LT.warning)
    
    def get_performance_stats(self):
        """렌더링 성능 통계"""
        if not self.render_times:
            return {}
            
        times = list(self.render_times)
        return {
            'avg_render_time': np.mean(times),
            'max_render_time': np.max(times),
            'render_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def close(self):
        """정리 함수"""
        self.stop_event.set()
        if self.fig is not None:
            plt.close(self.fig)

class SystemMonitor:
    """시스템 성능 모니터링"""
    def __init__(self):
        self.start_time = time.perf_counter()
        self.frame_times = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.last_stats_time = 0
        
        # psutil이 설치되어 있으면 사용
        self.psutil_available = False
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
            self.psutil_available = True
        except ImportError:
            printf("psutil not available - limited system monitoring", ptype=LT.warning)
    
    def update_frame_time(self, frame_time):
        """프레임 시간 업데이트"""
        self.frame_times.append(frame_time)
    
    def update_system_stats(self):
        """시스템 통계 업데이트"""
        if not self.psutil_available:
            return
            
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_mb)
        except Exception as e:
            printf(f"System stats update failed: {e}", ptype=LT.warning)
    
    def get_performance_report(self):
        """성능 리포트 생성"""
        current_time = time.perf_counter()
        uptime = current_time - self.start_time
        
        report = {
            'uptime': uptime,
            'total_frames': len(self.frame_times)
        }
        
        if self.frame_times:
            times = list(self.frame_times)
            report.update({
                'avg_fps': len(times) / sum(times) if sum(times) > 0 else 0,
                'avg_frame_time': np.mean(times),
                'max_frame_time': np.max(times),
                'min_frame_time': np.min(times)
            })
        
        if self.psutil_available and self.cpu_usage and self.memory_usage:
            report.update({
                'avg_cpu_usage': np.mean(list(self.cpu_usage)),
                'max_cpu_usage': np.max(list(self.cpu_usage)),
                'avg_memory_mb': np.mean(list(self.memory_usage)),
                'max_memory_mb': np.max(list(self.memory_usage))
            })
        
        return report
    
    def print_stats(self):
        """주기적 통계 출력"""
        current_time = time.perf_counter()
        stats_interval = CONFIG['processing']['update_intervals']['stats']
        
        if current_time - self.last_stats_time < stats_interval:
            return
            
        report = self.get_performance_report()
        
        printf("=== Performance Stats ===", ptype=LT.info)
        printf(f"Uptime: {report['uptime']:.1f}s", ptype=LT.info)
        printf(f"Total Frames: {report['total_frames']}", ptype=LT.info)
        
        if 'avg_fps' in report:
            printf(f"Average FPS: {report['avg_fps']:.1f}", ptype=LT.info)
            printf(f"Frame Time: avg={report['avg_frame_time']*1000:.1f}ms, max={report['max_frame_time']*1000:.1f}ms", ptype=LT.info)
        
        if 'avg_cpu_usage' in report:
            printf(f"CPU Usage: avg={report['avg_cpu_usage']:.1f}%, max={report['max_cpu_usage']:.1f}%", ptype=LT.info)
            printf(f"Memory: avg={report['avg_memory_mb']:.1f}MB, max={report['max_memory_mb']:.1f}MB", ptype=LT.info)
        
        self.last_stats_time = current_time

class OptimizedBallPlaceChecker:
    """최적화된 볼 위치 체커"""
    def __init__(self, zones: FieldZones):
        self.zones = zones
        self.flags = {
            "On Floor": False, "Hitted": False, "Thrower": False, "OutLined": False,
            "L In": False, "R In": False, "L Out": False, "R Out": False, "Running": False,
        }
        self.last_position = None
        self.last_result = None
        self.position_cache = {}
        self.cache_threshold = 0.1

    def check(self, bx: float, by: float) -> Optional[str]:
        """볼 위치 체크 - 고급 캐싱"""
        # 캐시 키 생성
        cache_key = (round(bx / self.cache_threshold), round(by / self.cache_threshold))
        
        # 캐시에서 확인
        if cache_key in self.position_cache:
            result = self.position_cache[cache_key]
            self._update_flags(result)
            return result
            
        # 새로운 위치 계산
        result = None
        for key in self.flags:
            self.flags[key] = False

        if self._in_box_fast(bx, by, self.zones.li):
            self.flags["L In"] = True
            result = "li"
        elif self._in_box_fast(bx, by, self.zones.ri):
            self.flags["R In"] = True
            result = "ri"
        elif self._in_box_fast(bx, by, self.zones.lo):
            self.flags["L Out"] = True
            result = "lo"
        elif self._in_box_fast(bx, by, self.zones.ro):
            self.flags["R Out"] = True
            result = "ro"
            
        # 캐시 업데이트 (크기 제한)
        if len(self.position_cache) < 1000:
            self.position_cache[cache_key] = result
            
        return result
    
    def _update_flags(self, result):
        """플래그 업데이트"""
        for key in self.flags:
            self.flags[key] = False
            
        if result == "li":
            self.flags["L In"] = True
        elif result == "ri":
            self.flags["R In"] = True
        elif result == "lo":
            self.flags["L Out"] = True
        elif result == "ro": 
            self.flags["R Out"] = True
    
    @staticmethod
    def _in_box_fast(x: float, y: float, box: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """최적화된 박스 내부 확인"""
        (xmin, ymin), (xmax, ymax) = box
        return xmin <= x <= xmax and ymin <= y <= ymax

def find_cameras_advanced():
    """고급 카메라 탐색 및 설정"""
    camera_count = int(input("Camera Count: "))
    available_cameras = {}
    selected_cameras = {}
    
    printf("Advanced camera search starting...", ptype=LT.info)
    
    search_range = CONFIG['camera']['search_range']
    
    # 병렬 카메라 체크
    import concurrent.futures
    
    def check_camera(i):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # 카메라 정보 수집
                    info = {
                        'index': i,
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'backend': cap.getBackendName()
                    }
                    cap.release()
                    return info
                cap.release()
        except:
            pass
        return None
    
    # 병렬 탐색
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(check_camera, i) for i in range(search_range)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                available_cameras[result['index']] = result
                printf(f"Camera {result['index']}: {result['width']}x{result['height']} @ {result['fps']}fps ({result['backend']})", ptype=LT.info)
    
    if len(available_cameras) == 0:
        printf("No cameras found!", ptype=LT.error)
        return {}
    
    printf(f"Found {len(available_cameras)} cameras", ptype=LT.info)
    
    # 자동 선택 (품질 기반)
    sorted_cameras = sorted(available_cameras.items(), 
                          key=lambda x: x[1]['width'] * x[1]['height'], 
                          reverse=True)
    
    for idx, (device_idx, info) in enumerate(sorted_cameras[:camera_count]):
        selected_cameras[idx] = device_idx
        printf(f"Auto-selected Camera {idx} -> Device {device_idx} ({info['width']}x{info['height']})", ptype=LT.info)
    
    return selected_cameras

def optimized_camera_thread_advanced(cam_id: int, device_id: int, stop_flag_ref, detector: ProfiledBallDetector):
    """고급 카메라 스레드"""
    printf(f"Starting advanced camera thread cam:{cam_id}", ptype=LT.info)

    try:
        cap = cv2.VideoCapture(device_id)
        
        # 고급 카메라 설정
        settings = [
            (cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH),
            (cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT),
            (cv2.CAP_PROP_FPS, TARGET_FPS),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
            (cv2.CAP_PROP_AUTOFOCUS, 0),  # 자동 포커스 비활성화
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 자동 노출 제한
        ]
        
        for prop, value in settings:
            try:
                cap.set(prop, value)
            except:
                pass  # 일부 속성은 카메라가 지원하지 않을 수 있음
        
        if not cap.isOpened():
            printf(f"Failed to open cam:{cam_id}", ptype=LT.error)
            streams[cam_id] = CamStream(None)
            return

        # 초기 통계 설정
        stream = CamStream(cap)
        stream.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'avg_fps': 0,
            'last_fps_calc': time.perf_counter()
        }
        streams[cam_id] = stream
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        while not stop_flag_ref[0]:
            frame_start = time.perf_counter()
            
            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    with streams[cam_id].lock:
                        if len(streams[cam_id].frames) >= BUFFER_SIZE:
                            streams[cam_id].stats['frames_dropped'] += 1
                        
                        streams[cam_id].frames.append(frame)
                        streams[cam_id].last_frame_time = frame_start
                        streams[cam_id].consecutive_failures = 0
                    
                    frame_count += 1
                    streams[cam_id].stats['frames_captured'] += 1
                    
                    # FPS 계산
                    frame_time = time.perf_counter() - frame_start
                    if frame_time > 0:
                        fps_counter.append(1.0 / frame_time)
                        if len(fps_counter) == 30:
                            streams[cam_id].stats['avg_fps'] = np.mean(fps_counter)
                    
                    # 주기적 통계 업데이트
                    if frame_count % 300 == 0:
                        printf(f"Camera {cam_id}: {frame_count} frames, avg FPS: {streams[cam_id].stats['avg_fps']:.1f}", ptype=LT.debug)
                else:
                    streams[cam_id].consecutive_failures += 1
                    if streams[cam_id].consecutive_failures > 30:
                        printf(f"Camera {cam_id}: Too many failures", ptype=LT.error)
                        break
                        
                # 적응형 대기 시간
                target_frame_time = 1.0 / TARGET_FPS
                actual_frame_time = time.perf_counter() - frame_start
                if actual_frame_time < target_frame_time:
                    time.sleep(target_frame_time - actual_frame_time)
                    
            except Exception as e:
                printf(f"Camera {cam_id} capture error: {e}", ptype=LT.error)
                time.sleep(0.1)

    except Exception as e:
        printf(f"Camera thread {cam_id} init failed: {e}", ptype=LT.error)
        streams[cam_id] = CamStream(None)
    finally:
        if 'cap' in locals():
            cap.release()
        printf(f"Advanced camera thread {cam_id} stopped", ptype=LT.info)

def build_point_grid_optimized() -> List[Tuple[float, float]]:
    """최적화된 격자 생성"""
    pdx = [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11]
    pdy = [7, 7, 7, 7, 4, 4, 4, 4, -4, -4, -4, -4, -7, -7, -7, -7]
    return list(zip(pdx, pdy))

def make_field_zones_optimized(point_list: List[Tuple[float, float]]) -> FieldZones:
    """최적화된 필드 영역 생성"""
    P = point_list
    
    def make_box(p1_idx, p2_idx):
        x1, y1 = P[p1_idx]
        x2, y2 = P[p2_idx] 
        return ((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)))

    return FieldZones(
        li=make_box(4, 0),
        ri=make_box(7, 3),
        lo=make_box(12, 15),
        ro=make_box(8, 11)
    )

# 전역 변수들
stop_flag_ref = [False]
streams: Dict[int, CamStream] = {}

def setup_logging():
    """간단하고 안정적인 로깅 시스템 설정"""
    log_config = CONFIG['logging']
    
    try:
        # 로그 디렉토리 생성
        log_file = Path(log_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 기본 로깅 설정
        logging.basicConfig(
            level=getattr(logging, log_config['level'], logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # 기존 설정 덮어쓰기
        )
        
        logger = logging.getLogger('BallTracker')
        logger.setLevel(getattr(logging, log_config['level'], logging.INFO))
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 파일 핸들러 (간단한 버전)
        try:
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)
        except Exception as e:
            printf(f"File logging failed: {e}. Using console only.", ptype=LT.warning)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        printf("Logging system initialized", ptype=LT.info)
        logger.info("Ball Tracker logging started")
        return logger
        
    except Exception as e:
        printf(f"Logging setup failed: {e}. Using basic print.", ptype=LT.warning)
        
        # 최후의 수단: 더미 로거
        class DummyLogger:
            def info(self, msg): printf(f"INFO: {msg}", ptype=LT.info)
            def warning(self, msg): printf(f"WARNING: {msg}", ptype=LT.warning)
            def error(self, msg): printf(f"ERROR: {msg}", ptype=LT.error)
            def debug(self, msg): printf(f"DEBUG: {msg}", ptype=LT.debug)
        
        return DummyLogger()

def create_performance_dashboard():
    """성능 대시보드 생성"""
    try:
        import tkinter as tk
        from tkinter import ttk
        
        class PerformanceDashboard:
            def __init__(self):
                self.root = tk.Tk()
                self.root.title("Ball Tracker Performance Dashboard")
                self.root.geometry("600x400")
                
                # 프레임 생성
                main_frame = ttk.Frame(self.root, padding="10")
                main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                
                # 성능 메트릭 라벨들
                self.labels = {}
                metrics = ['FPS', 'Detection Rate', 'CPU Usage', 'Memory Usage', 
                          'Camera Status', 'Triangulation Success']
                
                for i, metric in enumerate(metrics):
                    ttk.Label(main_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, pady=2)
                    self.labels[metric] = ttk.Label(main_frame, text="N/A")
                    self.labels[metric].grid(row=i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
                
                # 업데이트 버튼
                ttk.Button(main_frame, text="Refresh", command=self.update_metrics).grid(
                    row=len(metrics), column=0, columnspan=2, pady=10)
                
                self.monitor = None
                
            def set_monitor(self, monitor):
                self.monitor = monitor
                
            def update_metrics(self):
                if not self.monitor:
                    return
                    
                try:
                    report = self.monitor.get_performance_report()
                    
                    self.labels['FPS'].config(text=f"{report.get('avg_fps', 0):.1f}")
                    
                    if 'avg_cpu_usage' in report:
                        self.labels['CPU Usage'].config(text=f"{report['avg_cpu_usage']:.1f}%")
                        self.labels['Memory Usage'].config(text=f"{report['avg_memory_mb']:.1f} MB")
                    
                    # 카메라 상태
                    active_cams = sum(1 for s in streams.values() if s.cap is not None)
                    self.labels['Camera Status'].config(text=f"{active_cams} Active")
                    
                except Exception as e:
                    print(f"Dashboard update error: {e}")
                    
            def start(self):
                # 주기적 업데이트
                def auto_update():
                    self.update_metrics()
                    self.root.after(1000, auto_update)  # 1초마다
                
                auto_update()
                self.root.mainloop()
        
        return PerformanceDashboard()
        
    except ImportError:
        printf("tkinter not available - dashboard disabled", ptype=LT.warning)
        return None

def main():
    """고급 최적화가 적용된 메인 함수"""
    global streams
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("Advanced Ball Tracking System Starting")
    
    # 시스템 모니터 초기화
    monitor = SystemMonitor()
    
    # 성능 대시보드 (선택사항)
    dashboard = None
    if input("Enable performance dashboard? (y/N): ").lower() == 'y':
        dashboard = create_performance_dashboard()
        if dashboard:
            dashboard.set_monitor(monitor)
            # 별도 스레드에서 대시보드 실행
            dashboard_thread = Thread(target=dashboard.start, daemon=True)
            dashboard_thread.start()
    
    # 프로파일링 설정 개선
    enable_profiling = False
    
    # 방법 1: 환경변수 확인
    if os.getenv('PROFILE', 'False').lower() == 'true':
        enable_profiling = True
        printf("Profiling enabled via environment variable", ptype=LT.info)
    
    # 방법 2: 설정 파일 확인
    elif CONFIG.get('profiling', {}).get('enabled', False):
        enable_profiling = True
        printf("Profiling enabled via config file", ptype=LT.info)
    
    # 방법 3: 사용자에게 물어보기
    else:
        try:
            user_choice = input("Enable detailed profiling? (y/N): ").lower()
            if user_choice == 'y':
                enable_profiling = True
                printf("Profiling enabled by user choice", ptype=LT.info)
        except KeyboardInterrupt:
            pass
    
    main_profiler = None
    if enable_profiling:
        main_profiler = cProfile.Profile()
        main_profiler.enable()
        printf("Main profiling started", ptype=LT.info)
    
    try:
        printf("Advanced Ball Tracking System Starting", ptype=LT.info)
        
        # 1. 고급 카메라 설정
        camera_indices = find_cameras_advanced()
        
        if len(camera_indices) == 0:
            printf("No cameras found", ptype=LT.error)
            return
            
        printf(f"Selected cameras: {camera_indices}", ptype=LT.info)

        # 2. 볼 검출기 초기화
        detector = ProfiledBallDetector()
        
        # 3. 고급 카메라 스레드 시작
        threads = []
        for cam_id, dev_id in camera_indices.items():
            t = Thread(target=optimized_camera_thread_advanced, 
                      args=(cam_id, dev_id, stop_flag_ref, detector), daemon=True)
            t.start()
            threads.append(t)

        # 4. 카메라 준비 대기
        printf("Waiting for cameras...", ptype=LT.info)
        start_time = time.perf_counter()
        while len(streams) < len(camera_indices) and time.perf_counter() - start_time < 10.0:
            time.sleep(0.1)
            monitor.update_system_stats()

        active_cameras = sum(1 for s in streams.values() if s.cap is not None)
        printf(f"Active cameras: {active_cameras}", ptype=LT.info)

        if active_cameras == 0:
            printf("No active cameras!", ptype=LT.error)
            return

        # 5. 고급 캘리브레이션
        try:
            cam_configs = []
            for i in range(1, min(4, len(camera_indices) + 1)):
                if i in CONFIG['cameras']:
                    config = {"id": f"cam{i}"}
                    config.update(CONFIG['cameras'][i])
                    cam_configs.append(config)
            
            calibrate = CameraCalibration(cam_configs, FRAME_WIDTH, FRAME_HEIGHT, 800, 800)
            camera_params = calibrate.get_camera_params()
            printf("Advanced calibration completed", ptype=LT.info)
        except Exception as e:
            printf(f"Calibration failed: {e}", ptype=LT.error)
            return

        # 6. 고급 컴포넌트 초기화
        tracker = BallTracker3D(camera_params)
        
        # 데이터 구조 최적화
        history_size = CONFIG['processing']['position_history_size']
        position_history = deque(maxlen=history_size)
        
        point_list = build_point_grid_optimized()
        zones = make_field_zones_optimized(point_list)
        place_checker = OptimizedBallPlaceChecker(zones)
        
        # 고급 UI 및 애니메이션 초기화
        animate = AdvancedAnimationWrapper()
        
        try:
            interface = UserInterface()
            printf("Advanced interface initialized", ptype=LT.info)
        except:
            interface = None
            printf("Interface initialization failed - continuing without UI", ptype=LT.warning)
        
        # 7. 고급 메인 루프
        frame_interval = 1.0 / TARGET_FPS
        stats = {
            'detection_count': 0, 
            'triangulation_count': 0, 
            'loop_count': 0,
            'successful_triangulations': 0,
            'failed_triangulations': 0
        }
        
        printf("Starting advanced main loop", ptype=LT.info)
        printf("Controls: 'q'=quit, 'd'=debug, 'p'=plot, 's'=stats, 'r'=reset", ptype=LT.info)
        
        last_detection_log = 0
        
        while True:
            loop_start = time.perf_counter()
            stats['loop_count'] += 1

            # 시스템 모니터링
            monitor.update_frame_time(time.perf_counter() - loop_start)
            monitor.update_system_stats()
            
            # 주기적 통계 출력
            monitor.print_stats()

            # 고급 프레임 수집
            snapshot = {}
            for cam_id, stream in streams.items():
                if stream.cap is not None and stream.frames:
                    with stream.lock:
                        if stream.frames:
                            snapshot[cam_id] = stream.frames[-1].copy()

            if not snapshot:
                time.sleep(0.01)
                continue

            # 고급 볼 검출
            pts_2d = []
            cam_ids = []
            
            for cam_id, frame in snapshot.items():
                try:
                    # 프로파일된 검출 수행
                    pt, has_detection = detector.detect(frame, cam_id, stats['loop_count'])
                    
                    if pt is not None:
                        pts_2d.append(pt)
                        cam_ids.append(cam_id)
                        stats['detection_count'] += 1
                        
                        # 캐시 업데이트
                        streams[cam_id].detection_cache = pt
                        streams[cam_id].cache_time = time.perf_counter()
                        
                        # 향상된 시각적 피드백
                        radius = CONFIG['display']['circle_radius']
                        thickness = CONFIG['display']['line_thickness']
                        cv2.circle(frame, tuple(map(int, pt)), radius, (0, 255, 0), thickness)
                        
                        # 검출 정확도 표시
                        detection_stats = detector.get_stats()
                        if detection_stats:
                            info_text = f"Det FPS: {detection_stats.get('detection_fps', 0):.1f}"
                            cv2.putText(frame, info_text, (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # 상태 표시 업데이트
                    if stats['loop_count'] % 30 == 0:
                        cam_stats = streams[cam_id].stats
                        info = f"FPS:{cam_stats.get('avg_fps', 0):.1f} D:{stats['detection_count']} T:{stats['triangulation_count']}"
                        cv2.putText(frame, info, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imshow(f"CAM{cam_id}", frame)
                    
                except Exception as e:
                    if stats['loop_count'] % 200 == 0:  # 에러 로그 더 제한
                        printf(f"Detection error cam{cam_id}: {e}", ptype=LT.error)
                        logger.error(f"Detection error cam{cam_id}: {e}")

            # 고급 삼각측량
            if len(pts_2d) >= 2:
                try:
                    position_3d = tracker.triangulate_point(pts_2d, cam_ids)
                    
                    if position_3d is not None and not np.any(np.isnan(position_3d)) and not np.any(np.isinf(position_3d)):
                        stats['triangulation_count'] += 1
                        stats['successful_triangulations'] += 1
                        timestamp = time.perf_counter()
                        
                        # 상태 업데이트
                        state = tracker.update_state(position_3d, timestamp)
                        
                        if state.get('position') is not None:
                            position_entry = {
                                'timestamp': timestamp,
                                'position': tuple(state['position']),
                                'velocity': tuple(state.get('velocity', (0, 0, 0))),
                                'confidence': len(pts_2d) / len(camera_indices)  # 검출 신뢰도
                            }
                            position_history.append(position_entry)
                        
                        # UI 업데이트 (빈도 제한)
                        if stats['loop_count'] % 15 == 0:
                            bx, by = state["position"][0], state["position"][1]
                            zone = place_checker.check(bx, by)
                            
                            if interface:
                                try:
                                    interface.update(place_checker.flags)
                                except Exception as e:
                                    if stats['loop_count'] % 300 == 0:
                                        printf(f"UI update error: {e}", ptype=LT.warning)
                        
                        # 고급 애니메이션 데이터 업데이트
                        if stats['loop_count'] % 20 == 0:  # 더 낮은 빈도
                            pl_data = {p['timestamp']: p['position'] for p in position_history}
                            animate.update_data(pl_data)
                        
                        # 상세 로깅 (주기적)
                        if time.perf_counter() - last_detection_log > 2.0:
                            printf(f"Position: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}), "
                                  f"Confidence: {position_entry['confidence']:.2f}", ptype=LT.success)
                            last_detection_log = time.perf_counter()
                        
                    else:
                        stats['failed_triangulations'] += 1
                        
                except Exception as e:
                    stats['failed_triangulations'] += 1
                    if stats['loop_count'] % 200 == 0:
                        printf(f"Triangulation error: {e}", ptype=LT.error)
                        logger.error(f"Triangulation error: {e}")

            # 고급 애니메이션 처리 (메인 스레드)
            if stats['loop_count'] % 15 == 0:
                animate.process_updates()

            # 고급 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # 현재 데이터로 플롯 업데이트
                pl_data = {p['timestamp']: p['position'] for p in position_history}
                animate._update_plot_advanced({'pl': pl_data})
            elif key == ord('s'):
                # 상세 통계 출력
                detection_stats = detector.get_stats()
                animation_stats = animate.get_performance_stats()
                system_stats = monitor.get_performance_report()
                
                printf("=== Detailed Statistics ===", ptype=LT.info)
                printf(f"Detection: {detection_stats}", ptype=LT.info)
                printf(f"Animation: {animation_stats}", ptype=LT.info)
                printf(f"System: {system_stats}", ptype=LT.info)
                printf(f"Success Rate: {stats['successful_triangulations']}/{stats['successful_triangulations']+stats['failed_triangulations']}", ptype=LT.info)
            elif key == ord('r'):
                # 통계 리셋
                stats = {k: 0 for k in stats.keys()}
                position_history.clear()
                printf("Statistics reset", ptype=LT.info)

            # 적응형 FPS 제한
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            elif elapsed > frame_interval * 2:
                # 프레임 드롭 감지
                printf(f"Frame drop detected: {elapsed*1000:.1f}ms", ptype=LT.warning)

    except KeyboardInterrupt:
        printf("Terminated by user", ptype=LT.info)
        logger.info("System terminated by user")
    except Exception as e:
        printf(f"Main loop error: {e}", ptype=LT.error)
        logger.error(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 고급 정리 프로세스
        printf("Starting advanced cleanup...", ptype=LT.info)
        stop_flag_ref[0] = True
        time.sleep(0.3)
        
        # 프로파일 결과 저장
        if enable_profiling and main_profiler:
            main_profiler.disable()
            main_profiler.dump_stats("main_profile.prof")
            printf("Main profile saved", ptype=LT.info)
        
        # 검출기 프로파일 저장
        detector.save_profile("detection_profile.prof")
        
        # 애니메이션 정리
        animate.close()
        cv2.destroyAllWindows()
        
        # 최종 통계
        printf("=== Final Advanced Statistics ===", ptype=LT.info)
        printf(f"Total Loops: {stats['loop_count']}", ptype=LT.info)
        printf(f"Total Detections: {stats['detection_count']}", ptype=LT.info)
        printf(f"Total Triangulations: {stats['triangulation_count']}", ptype=LT.info)
        printf(f"Success Rate: {stats['successful_triangulations']}/{stats['successful_triangulations']+stats['failed_triangulations']} ({100*stats['successful_triangulations']/(stats['successful_triangulations']+stats['failed_triangulations']) if (stats['successful_triangulations']+stats['failed_triangulations']) > 0 else 0:.1f}%)", ptype=LT.info)
        
        if stats['loop_count'] > 0:
            printf(f"Detection Rate: {stats['detection_count']/stats['loop_count']*100:.1f}%", ptype=LT.info)
            printf(f"Average Loop Time: {monitor.get_performance_report().get('avg_frame_time', 0)*1000:.1f}ms", ptype=LT.info)
        
        # 성능 보고서를 파일로 저장
        try:
            performance_report = monitor.get_performance_report()
            performance_report.update({
                'final_stats': stats,
                'detection_stats': detector.get_stats(),
                'animation_stats': animate.get_performance_stats()
            })
            
            with open('performance_report.yaml', 'w') as f:
                yaml.dump(performance_report, f, default_flow_style=False)
            printf("Performance report saved to performance_report.yaml", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to save performance report: {e}", ptype=LT.warning)
        
        logger.info("Advanced Ball Tracking System shutdown complete")

if __name__ == "__main__":
    main()