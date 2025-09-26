# core/managers/camera_manager.py
"""
카메라 관리자 클래스
기존의 find_cameras_advanced, optimized_camera_thread_advanced 함수들을 통합 관리
"""

import cv2
import time
import concurrent.futures
from threading import Thread, Lock
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.models.camera_stream import CamStream
from classes.printing import printf, LT


class CameraManager:
    """카메라 검색, 설정, 스트림 관리를 담당하는 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_config = config['camera']
        
        # 카메라 설정값
        self.frame_width = self.camera_config['width']
        self.frame_height = self.camera_config['height']
        self.target_fps = self.camera_config['fps']
        self.buffer_size = self.camera_config['buffer_size']
        self.search_range = self.camera_config['search_range']
        
        # 카메라 스트림들
        self.streams: Dict[int, CamStream] = {}
        self.stop_flag_ref = [False]
        self.camera_threads: List[Thread] = []
        self.selected_cameras: Dict[int, int] = {}
        
    def find_available_cameras(self) -> Dict[int, Dict[str, Any]]:
        """사용 가능한 카메라들을 병렬로 검색"""
        printf("Advanced camera search starting...", LT.info)
        
        available_cameras = {}
        
        def check_camera(i):
            """개별 카메라 확인"""
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
            except Exception as e:
                printf(f"Error checking camera {i}: {e}", LT.debug)
            return None
        
        # 병렬 탐색
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(check_camera, i) for i in range(self.search_range)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    available_cameras[result['index']] = result
                    printf(f"Camera {result['index']}: {result['width']}x{result['height']} @ "
                          f"{result['fps']}fps ({result['backend']})", LT.info)
        
        if len(available_cameras) == 0:
            printf("No cameras found!", LT.error)
            
        printf(f"Found {len(available_cameras)} cameras", LT.info)
        return available_cameras
    
    def select_cameras(self, available_cameras: Dict[int, Dict[str, Any]], 
                      camera_count: int) -> Dict[int, int]:
        """최적의 카메라들을 자동 선택"""
        if not available_cameras:
            return {}
        
        # 품질 기반 정렬 (해상도 기준)
        sorted_cameras = sorted(
            available_cameras.items(), 
            key=lambda x: x[1]['width'] * x[1]['height'], 
            reverse=True
        )
        
        selected = {}
        for idx, (device_idx, info) in enumerate(sorted_cameras[:camera_count]):
            selected[idx] = device_idx
            printf(f"Auto-selected Camera {idx} -> Device {device_idx} "
                  f"({info['width']}x{info['height']})", LT.info)
        
        return selected
    
    def initialize_cameras(self, camera_count: int = None) -> bool:
        """카메라 초기화 및 선택"""
        if camera_count is None:
            try:
                camera_count = int(input("Camera Count: "))
            except (ValueError, KeyboardInterrupt):
                printf("Invalid camera count input", LT.error)
                return False
        
        # 사용 가능한 카메라 검색
        available_cameras = self.find_available_cameras()
        
        if not available_cameras:
            return False
        
        # 카메라 선택
        self.selected_cameras = self.select_cameras(available_cameras, camera_count)
        
        if not self.selected_cameras:
            printf("No cameras selected", LT.error)
            return False
        
        printf(f"Selected cameras: {self.selected_cameras}", LT.info)
        return True
    
    def _setup_camera(self, cap: cv2.VideoCapture) -> bool:
        """개별 카메라 설정"""
        settings = [
            (cv2.CAP_PROP_FRAME_WIDTH, self.frame_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height),
            (cv2.CAP_PROP_FPS, self.target_fps),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
            (cv2.CAP_PROP_AUTOFOCUS, 0),  # 자동 포커스 비활성화
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 자동 노출 제한
        ]
        
        success_count = 0
        for prop, value in settings:
            try:
                if cap.set(prop, value):
                    success_count += 1
            except Exception:
                pass  # 일부 속성은 카메라가 지원하지 않을 수 있음
        
        return success_count > 0
    
    def _camera_thread(self, cam_id: int, device_id: int):
        """개별 카메라 스레드"""
        printf(f"Starting camera thread cam:{cam_id}", LT.info)
        
        try:
            cap = cv2.VideoCapture(device_id)
            
            # 카메라 설정
            self._setup_camera(cap)
            
            if not cap.isOpened():
                printf(f"Failed to open cam:{cam_id}", LT.error)
                self.streams[cam_id] = CamStream(None)
                return
            
            # 초기 통계 설정
            stream = CamStream(cap)
            stream.stats = {
                'frames_captured': 0,
                'frames_dropped': 0,
                'avg_fps': 0,
                'last_fps_calc': time.perf_counter()
            }
            self.streams[cam_id] = stream
            
            frame_count = 0
            fps_counter = deque(maxlen=30)
            
            while not self.stop_flag_ref[0]:
                frame_start = time.perf_counter()
                
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        with self.streams[cam_id].lock:
                            if len(self.streams[cam_id].frames) >= self.buffer_size:
                                self.streams[cam_id].stats['frames_dropped'] += 1
                            
                            self.streams[cam_id].frames.append(frame)
                            self.streams[cam_id].last_frame_time = frame_start
                            self.streams[cam_id].consecutive_failures = 0
                        
                        frame_count += 1
                        self.streams[cam_id].stats['frames_captured'] += 1
                        
                        # FPS 계산
                        frame_time = time.perf_counter() - frame_start
                        if frame_time > 0:
                            fps_counter.append(1.0 / frame_time)
                            if len(fps_counter) == 30:
                                self.streams[cam_id].stats['avg_fps'] = np.mean(fps_counter)
                        
                        # 주기적 통계 업데이트
                        if frame_count % 300 == 0:
                            printf(f"Camera {cam_id}: {frame_count} frames, "
                                  f"avg FPS: {self.streams[cam_id].stats['avg_fps']:.1f}", 
                                  LT.debug)
                    else:
                        self.streams[cam_id].consecutive_failures += 1
                        if self.streams[cam_id].consecutive_failures > 30:
                            printf(f"Camera {cam_id}: Too many failures", LT.error)
                            break
                            
                    # 적응형 대기 시간
                    target_frame_time = 1.0 / self.target_fps
                    actual_frame_time = time.perf_counter() - frame_start
                    if actual_frame_time < target_frame_time:
                        time.sleep(target_frame_time - actual_frame_time)
                        
                except Exception as e:
                    printf(f"Camera {cam_id} capture error: {e}", LT.error)
                    time.sleep(0.1)
        
        except Exception as e:
            printf(f"Camera thread {cam_id} init failed: {e}", LT.error)
            self.streams[cam_id] = CamStream(None)
        finally:
            if 'cap' in locals():
                cap.release()
            printf(f"Camera thread {cam_id} stopped", LT.info)
    
    def start_camera_threads(self) -> bool:
        """카메라 스레드들 시작"""
        if not self.selected_cameras:
            printf("No cameras selected", LT.error)
            return False
        
        printf("Starting camera threads...", LT.info)
        
        for cam_id, device_id in self.selected_cameras.items():
            thread = Thread(
                target=self._camera_thread,
                args=(cam_id, device_id),
                daemon=True
            )
            thread.start()
            self.camera_threads.append(thread)
        
        # 카메라 준비 대기
        printf("Waiting for cameras...", LT.info)
        start_time = time.perf_counter()
        while (len(self.streams) < len(self.selected_cameras) and 
               time.perf_counter() - start_time < 10.0):
            time.sleep(0.1)
        
        active_cameras = sum(1 for s in self.streams.values() if s.cap is not None)
        printf(f"Active cameras: {active_cameras}", LT.info)
        
        return active_cameras > 0
    
    def get_frame_snapshot(self) -> Dict[int, np.ndarray]:
        """모든 활성 카메라에서 최신 프레임 수집"""
        snapshot = {}
        for cam_id, stream in self.streams.items():
            if stream.cap is not None and stream.frames:
                with stream.lock:
                    if stream.frames:
                        snapshot[cam_id] = stream.frames[-1].copy()
        return snapshot
    
    def get_camera_stats(self) -> Dict[int, Dict[str, Any]]:
        """카메라 통계 반환"""
        stats = {}
        for cam_id, stream in self.streams.items():
            if stream.cap is not None:
                stats[cam_id] = stream.stats.copy()
        return stats
    
    def stop_cameras(self):
        """모든 카메라 스레드 정지"""
        printf("Stopping camera threads...", LT.info)
        self.stop_flag_ref[0] = True
        
        # 스레드 종료 대기
        for thread in self.camera_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # 카메라 리소스 해제
        for stream in self.streams.values():
            if stream.cap is not None:
                stream.cap.release()
        
        self.streams.clear()
        printf("All camera threads stopped", LT.info)