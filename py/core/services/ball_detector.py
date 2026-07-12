# core/services/ball_detector.py

import cv2
import time
import cProfile
import pstats
import numpy as np
import os
from collections import deque
from typing import Dict, Tuple, Optional, Any
from contextlib import contextmanager

from utils.printing import printf, LT


class DetectionConfig:
    """검출 설정 캐싱 클래스"""
    
    # 기본값 상수
    DEFAULT_DETECTION_INTERVAL = 3
    DEFAULT_MIN_CONTOUR_AREA = 30
    DEFAULT_MORPHOLOGY_KERNEL_SIZE = 3
    DEFAULT_RESIZE_SCALE = 0.5
    DEFAULT_HSV_LOWER = [0, 50, 50]
    DEFAULT_HSV_UPPER = [15, 255, 255]
    DEFAULT_DNN_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_BALL_CLASS_ID = 0
    
    def __init__(self, config: Dict[str, Any]):
        # 검출 간격
        self.detection_interval = config.get('detection_interval', self.DEFAULT_DETECTION_INTERVAL)
        
        # HSV 색상 범위
        self.hsv_lower = np.array(config.get('hsv_lower', self.DEFAULT_HSV_LOWER))
        self.hsv_upper = np.array(config.get('hsv_upper', self.DEFAULT_HSV_UPPER))
        
        # 컨투어 필터링
        self.min_contour_area = config.get('min_contour_area', self.DEFAULT_MIN_CONTOUR_AREA)
        
        # 형태학적 연산
        self.morphology_kernel_size = config.get(
            'morphology_kernel_size', 
            self.DEFAULT_MORPHOLOGY_KERNEL_SIZE
        )
        
        # 프레임 크기 조정
        self.resize_scale = config.get('resize_scale', self.DEFAULT_RESIZE_SCALE)
        self.scale_factor = int(1 / self.resize_scale)
        
        # DNN 설정
        self.enable_gpu = config.get('enable_gpu', False)
        self.model_path = config.get('model_path', 'ball_detection.onnx')
        self.dnn_confidence_threshold = config.get(
            'dnn_confidence_threshold',
            self.DEFAULT_DNN_CONFIDENCE_THRESHOLD
        )
        self.ball_class_id = config.get('ball_class_id', self.DEFAULT_BALL_CLASS_ID)
        
        # 프로파일링
        self.profiling_enabled = config.get('profiling_enabled', False)
        
        # 형태학적 커널 미리 생성 (캐싱)
        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )


class BallDetectorService:
    """프로파일링이 포함된 볼 검출 서비스"""
    
    DETECTION_TIMES_MAXLEN = 100
    
    def __init__(self, detection_config: Dict[str, Any]):
        # 설정 캐싱
        self.config = DetectionConfig(detection_config)
        
        # 검출 통계
        self.detection_times = deque(maxlen=self.DETECTION_TIMES_MAXLEN)
        
        # GPU/DNN 설정
        self.gpu_available = False
        self.net = None
        
        # 초기화
        self._initialize_detection_backend()
        self._initialize_profiling()
    
    def _initialize_detection_backend(self):
        """검출 백엔드 초기화 (내부 헬퍼)"""
        if self.config.enable_gpu:
            self._setup_gpu_detection()
        
        self._log_cuda_info()
    
    def _log_cuda_info(self):
        """CUDA 정보 로깅 (내부 헬퍼)"""
        try:
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_device_count > 0:
                printf(f"CUDA devices available: {cuda_device_count}", ptype=LT.info)
        except Exception:
            pass  # CUDA 정보 가져오기 실패는 무시
    
    def _initialize_profiling(self):
        """프로파일링 초기화 (내부 헬퍼)"""
        self.enable_profiling = self._should_enable_profiling()
        self.profiler = None
        
        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            printf("Ball detector profiling enabled", ptype=LT.info)
    
    def _should_enable_profiling(self) -> bool:
        """프로파일링 활성화 여부 판단 (내부 헬퍼)"""
        # 환경변수 우선
        if os.getenv('PROFILE', 'False').lower() == 'true':
            return True
        
        # 설정 파일
        return self.config.profiling_enabled
    
    def _setup_gpu_detection(self):
        """GPU 가속 볼 검출 설정"""
        if not os.path.exists(self.config.model_path):
            printf(f"Model file not found: {self.config.model_path}", ptype=LT.warning)
            return
        
        try:
            self.net = cv2.dnn.readNet(self.config.model_path)
            
            if self._try_enable_cuda():
                self.gpu_available = True
                printf("GPU acceleration enabled for ball detection", ptype=LT.info)
            else:
                printf("CUDA not available, using CPU", ptype=LT.warning)
                
        except Exception as e:
            printf(f"Failed to setup GPU detection: {e}", ptype=LT.warning)
    
    def _try_enable_cuda(self) -> bool:
        """CUDA 활성화 시도 (내부 헬퍼)"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                return True
        except Exception:
            pass
        return False
    
    @contextmanager
    def _profiling_context(self):
        """프로파일링 컨텍스트 매니저"""
        if self.enable_profiling and self.profiler:
            self.profiler.enable()
        
        try:
            yield
        finally:
            if self.enable_profiling and self.profiler:
                self.profiler.disable()
    
    def detect(
        self, 
        frame: np.ndarray, 
        cam_id: int, 
        frame_count: int
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """통합 볼 검출 함수"""
        with self._profiling_context():
            if self.net is not None:
                return self.detect_with_dnn(frame)
            else:
                return self.detect_traditional(frame, cam_id, frame_count)
    
    def detect_with_dnn(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """DNN을 사용한 볼 검출"""
        if self.net is None:
            return None, False
        
        try:
            blob = cv2.dnn.blobFromImage(
                frame, 
                1/255.0, 
                (416, 416), 
                swapRB=True, 
                crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            return self._process_dnn_outputs(outputs, frame.shape)
            
        except Exception as e:
            printf(f"DNN detection failed: {e}", ptype=LT.error)
            return None, False
    
    def _process_dnn_outputs(
        self, 
        outputs: np.ndarray, 
        frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """DNN 출력 처리 (내부 헬퍼)"""
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if (confidence > self.config.dnn_confidence_threshold and 
                    class_id == self.config.ball_class_id):
                    
                    center_x = int(detection[0] * frame_shape[1])
                    center_y = int(detection[1] * frame_shape[0])
                    return (center_x, center_y), True
        
        return None, False
    
    def detect_traditional(
        self, 
        frame: np.ndarray, 
        cam_id: int, 
        frame_count: int
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """전통적인 컴퓨터 비전을 사용한 볼 검출"""
        if not self._is_valid_frame(frame):
            return None, False
        
        if not self._should_detect_frame(frame_count):
            return None, False
        
        start_time = time.perf_counter()
        
        try:
            result = self._perform_traditional_detection(frame)
            
            # 검출 시간 기록
            if result[1]:  # 검출 성공 시
                detection_time = time.perf_counter() - start_time
                self.detection_times.append(detection_time)
            
            return result
            
        except Exception as e:
            printf(f"Traditional detection error cam{cam_id}: {e}", ptype=LT.error)
            return None, False
    
    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """유효한 프레임인지 확인 (내부 헬퍼)"""
        return frame is not None and frame.size > 0
    
    def _should_detect_frame(self, frame_count: int) -> bool:
        """프레임 검출 여부 판단 (내부 헬퍼)"""
        return frame_count % self.config.detection_interval == 0
    
    def _perform_traditional_detection(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """전통적 검출 수행 (내부 헬퍼)"""
        # 1. 프레임 전처리
        processed_frame = self._preprocess_frame(frame)
        
        # 2. 마스크 생성
        mask = self._create_color_mask(processed_frame)
        
        # 3. 형태학적 연산
        mask = self._apply_morphology(mask)
        
        # 4. 컨투어 처리
        return self._process_contours(mask)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 전처리 (내부 헬퍼)"""
        # 크기 조정
        new_width = int(frame.shape[1] * self.config.resize_scale)
        new_height = int(frame.shape[0] * self.config.resize_scale)
        small_frame = cv2.resize(frame, (new_width, new_height))
        
        # HSV 변환
        return cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    def _create_color_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """색상 마스크 생성 (내부 헬퍼)"""
        return cv2.inRange(hsv_frame, self.config.hsv_lower, self.config.hsv_upper)
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """형태학적 연산 적용 (내부 헬퍼)"""
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.config.morphology_kernel)
    
    def _process_contours(
        self, 
        mask: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """컨투어 처리 (내부 헬퍼)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, False
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area <= self.config.min_contour_area:
            return None, False
        
        return self._calculate_contour_center(largest_contour)
    
    def _calculate_contour_center(
        self, 
        contour: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """컨투어 중심점 계산 (내부 헬퍼)"""
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return None, False
        
        # 원래 크기로 좌표 변환
        cx = int(M["m10"] / M["m00"]) * self.config.scale_factor
        cy = int(M["m01"] / M["m00"]) * self.config.scale_factor
        
        return (cx, cy), True
    
    def get_stats(self) -> Dict[str, float]:
        """검출 성능 통계"""
        if not self.detection_times:
            return {}
        
        times_array = np.array(self.detection_times)
        mean_time = times_array.mean()
        
        return {
            'avg_detection_time': mean_time,
            'max_detection_time': times_array.max(),
            'min_detection_time': times_array.min(),
            'detection_fps': 1.0 / mean_time if mean_time > 0 else 0
        }
    
    def save_profile(self, filename: str = "detection_profile.prof"):
        """프로파일 결과 저장"""
        if not (self.enable_profiling and self.profiler):
            return
        
        try:
            self._save_profile_data(filename)
            self._save_profile_report(filename)
            printf(f"Detection profile saved to {filename}", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to save profile: {e}", ptype=LT.warning)
    
    def _save_profile_data(self, filename: str):
        """프로파일 데이터 저장 (내부 헬퍼)"""
        self.profiler.dump_stats(filename)
    
    def _save_profile_report(self, filename: str):
        """프로파일 텍스트 리포트 저장 (내부 헬퍼)"""
        report_filename = filename.replace('.prof', '.txt')
        
        with open(report_filename, 'w') as f:
            stats = pstats.Stats(self.profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()