# core/services/ball_detector.py
"""
볼 검출 서비스 클래스
기존 ProfiledBallDetector 클래스를 리팩터링하여 서비스로 분리
"""

import cv2
import time
import cProfile
import pstats
import numpy as np
import os
from collections import deque
from typing import Dict, Tuple, Optional, Any

from classes.printing import printf, LT


class BallDetectorService:
    """프로파일링이 포함된 볼 검출 서비스"""
    
    def __init__(self, detection_config: Dict[str, Any]):
        self.detection_config = detection_config
        self.detection_times = deque(maxlen=100)
        self.gpu_available = False
        self.net = None
        self.profiler = None
        
        # GPU 가속 설정 시도
        if detection_config.get('enable_gpu', False):
            self._setup_gpu_detection()
        
        # OpenCV DNN 백엔드 최적화
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            printf(f"CUDA devices available: {cv2.cuda.getCudaEnabledDeviceCount()}", LT.info)
        
        # 프로파일링 설정
        self.enable_profiling = self._setup_profiling()
        
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            printf("Ball detector profiling enabled", LT.info)
    
    def _setup_profiling(self) -> bool:
        """프로파일링 설정"""
        # 방법 1: 환경변수 사용
        if os.getenv('PROFILE', 'False').lower() == 'true':
            return True
        
        # 방법 2: 설정 파일에서 읽기
        if self.detection_config.get('profiling_enabled', False):
            return True
        
        return False
    
    def _setup_gpu_detection(self):
        """GPU 가속 볼 검출 설정"""
        model_path = self.detection_config.get('model_path', 'ball_detection.onnx')
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNet(model_path)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.gpu_available = True
                    printf("GPU acceleration enabled for ball detection", LT.info)
                else:
                    printf("CUDA not available, using CPU", LT.warning)
            except Exception as e:
                printf(f"Failed to setup GPU detection: {e}", LT.warning)
    
    def detect_with_dnn(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], bool]:
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
            printf(f"DNN detection failed: {e}", LT.error)
            return None, False
    
    def detect_traditional(self, frame: np.ndarray, cam_id: int, frame_count: int) -> Tuple[Optional[Tuple[int, int]], bool]:
        """전통적인 컴퓨터 비전을 사용한 볼 검출"""
        if frame is None or frame.size == 0:
            return None, False
        
        # 프레임 스킵 (설정된 간격마다 검출)
        detection_interval = self.detection_config.get('detection_interval', 3)
        if frame_count % detection_interval != 0:
            return None, False
        
        start_time = time.perf_counter()
        
        try:
            # 프레임 크기 조정 (성능 향상)
            small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            
            # HSV 변환
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # 설정된 색상 범위로 마스크 생성
            lower_color = np.array(self.detection_config.get('hsv_lower', [0, 50, 50]))
            upper_color = np.array(self.detection_config.get('hsv_upper', [15, 255, 255]))
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # 형태학적 연산
            kernel_size = self.detection_config.get('morphology_kernel_size', 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                min_area = self.detection_config.get('min_contour_area', 30)
                if area > min_area:
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
            printf(f"Traditional detection error cam{cam_id}: {e}", LT.error)
            return None, False
    
    def detect(self, frame: np.ndarray, cam_id: int, frame_count: int) -> Tuple[Optional[Tuple[int, int]], bool]:
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
    
    def get_stats(self) -> Dict[str, float]:
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
    
    def save_profile(self, filename: str = "detection_profile.prof"):
        """프로파일 결과 저장"""
        if self.enable_profiling and self.profiler:
            self.profiler.dump_stats(filename)
            
            # 텍스트 리포트도 생성
            with open(filename.replace('.prof', '.txt'), 'w') as f:
                stats = pstats.Stats(self.profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(file=f)
            
            printf(f"Detection profile saved to {filename}", LT.info)