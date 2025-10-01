# core/services/ball_detector.py

import cv2
import time
import cProfile
import pstats
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
from contextlib import contextmanager
from collections import deque
from pathlib import Path

from utils.printing import printf, LT


@dataclass
class DetectionConfig:
    """볼 검출 설정"""
    detection_interval: int = 3
    hsv_lower: tuple[int, int, int] = (0, 50, 50)
    hsv_upper: tuple[int, int, int] = (15, 255, 255)
    min_contour_area: int = 30
    morphology_kernel_size: int = 3
    resize_scale: float = 0.5
    dnn_confidence_threshold: float = 0.5
    ball_class_id: int = 0
    enable_gpu: bool = False
    model_path: Optional[Path] = None
    profiling_enabled: bool = False


@dataclass
class DetectionResult:
    """검출 결과"""
    position: Optional[tuple[int, int]]
    detected: bool
    confidence: float = 0.0
    detection_time: float = 0.0
    
    @property
    def x(self) -> Optional[int]:
        return self.position[0] if self.position else None
    
    @property
    def y(self) -> Optional[int]:
        return self.position[1] if self.position else None


class DetectionStrategy(ABC):
    """볼 검출 전략 인터페이스"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """프레임에서 볼 검출"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """검출 방법 사용 가능 여부"""
        pass


class DNNDetectionStrategy(DetectionStrategy):
    """DNN 기반 볼 검출"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.net = None
        self._available = False
        self._setup_network()
    
    def _setup_network(self) -> None:
        """네트워크 초기화"""
        if not self.config.model_path or not self.config.model_path.exists():
            printf(f"Model not found: {self.config.model_path}", ptype=LT.warning)
            return
        
        try:
            self.net = cv2.dnn.readNet(str(self.config.model_path))
            
            if self.config.enable_gpu and self._is_cuda_available():
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                printf("GPU acceleration enabled", ptype=LT.info)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self._available = True
            
        except cv2.error as e:
            printf(f"DNN setup failed: {e}", ptype=LT.error)
            self.net = None
    
    @staticmethod
    def _is_cuda_available() -> bool:
        """CUDA 사용 가능 여부"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """DNN으로 볼 검출"""
        if not self.is_available():
            return DetectionResult(None, False)
        
        start_time = time.perf_counter()
        
        try:
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416), 
                swapRB=True, crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            best_detection = self._parse_detections(outputs, frame.shape)
            detection_time = time.perf_counter() - start_time
            
            if best_detection:
                return DetectionResult(
                    position=best_detection['position'],
                    detected=True,
                    confidence=best_detection['confidence'],
                    detection_time=detection_time
                )
            
            return DetectionResult(None, False, detection_time=detection_time)
            
        except (cv2.error, ValueError) as e:
            printf(f"DNN detection error: {e}", ptype=LT.error)
            return DetectionResult(None, False)
    
    def _parse_detections(self, outputs: np.ndarray, frame_shape: tuple) -> Optional[dict]:
        """검출 결과 파싱"""
        best_confidence = 0
        best_detection = None
        
        for output in outputs:
            for detection in output:
                if len(detection) < 6:
                    continue
                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                if (class_id == self.config.ball_class_id and 
                    confidence > self.config.dnn_confidence_threshold and
                    confidence > best_confidence):
                    
                    center_x = int(detection[0] * frame_shape[1])
                    center_y = int(detection[1] * frame_shape[0])
                    
                    best_detection = {
                        'position': (center_x, center_y),
                        'confidence': confidence
                    }
                    best_confidence = confidence
        
        return best_detection
    
    def is_available(self) -> bool:
        return self._available and self.net is not None


class TraditionalDetectionStrategy(DetectionStrategy):
    """전통적인 컴퓨터 비전 기반 볼 검출"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_count = 0
        
        # 재사용 가능한 버퍼 (메모리 효율)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (config.morphology_kernel_size, config.morphology_kernel_size)
        )
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """전통적 방법으로 볼 검출"""
        self.frame_count += 1
        
        # 프레임 스킵
        if self.frame_count % self.config.detection_interval != 0:
            return DetectionResult(None, False)
        
        if not self._validate_frame(frame):
            return DetectionResult(None, False)
        
        start_time = time.perf_counter()
        
        try:
            # 프레임 전처리
            small_frame = self._resize_frame(frame)
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # 색상 기반 마스크
            mask = cv2.inRange(
                hsv,
                np.array(self.config.hsv_lower),
                np.array(self.config.hsv_upper)
            )
            
            # 노이즈 제거
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            position = self._find_ball_position(contours, small_frame.shape)
            detection_time = time.perf_counter() - start_time
            
            if position:
                # 원본 크기로 스케일링
                scale = 1.0 / self.config.resize_scale
                scaled_position = (
                    int(position[0] * scale),
                    int(position[1] * scale)
                )
                return DetectionResult(
                    scaled_position, True, 
                    detection_time=detection_time
                )
            
            return DetectionResult(None, False, detection_time=detection_time)
            
        except (cv2.error, ZeroDivisionError) as e:
            printf(f"Traditional detection error: {e}", ptype=LT.error)
            return DetectionResult(None, False)
    
    def _validate_frame(self, frame: np.ndarray) -> bool:
        """프레임 유효성 검사"""
        return (frame is not None and 
                frame.size > 0 and 
                len(frame.shape) == 3)
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """프레임 리사이즈"""
        if self.config.resize_scale == 1.0:
            return frame
        
        new_width = int(frame.shape[1] * self.config.resize_scale)
        new_height = int(frame.shape[0] * self.config.resize_scale)
        return cv2.resize(frame, (new_width, new_height))
    
    def _find_ball_position(self, contours, frame_shape) -> Optional[tuple[int, int]]:
        """컨투어에서 볼 위치 찾기"""
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area <= self.config.min_contour_area:
            return None
        
        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            return None
        
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        return (center_x, center_y)
    
    def is_available(self) -> bool:
        return True


class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self, window_size: int = 100):
        self.detection_times = deque(maxlen=window_size)
    
    def record(self, detection_time: float) -> None:
        """검출 시간 기록"""
        self.detection_times.append(detection_time)
    
    def get_stats(self) -> dict[str, float]:
        """통계 반환"""
        if not self.detection_times:
            return {}
        
        times = np.array(list(self.detection_times))
        return {
            'avg_detection_time': float(np.mean(times)),
            'max_detection_time': float(np.max(times)),
            'min_detection_time': float(np.min(times)),
            'std_detection_time': float(np.std(times)),
            'detection_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }


@contextmanager
def profiling_context(profiler: Optional[cProfile.Profile]):
    """프로파일링 컨텍스트 매니저"""
    if profiler:
        profiler.enable()
    try:
        yield
    finally:
        if profiler:
            profiler.disable()


class BallDetectorService:
    """볼 검출 서비스 (개선 버전)"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.profiler = cProfile.Profile() if config.profiling_enabled else None
        
        # 검출 전략 초기화 (우선순위: DNN -> Traditional)
        self.strategies = self._initialize_strategies()
        self.current_strategy = self._select_strategy()
        
        if config.profiling_enabled:
            printf("Profiling enabled", ptype=LT.info)
    
    def _initialize_strategies(self) -> list[DetectionStrategy]:
        """검출 전략 초기화"""
        strategies = []
        
        # DNN 전략
        if self.config.model_path:
            dnn_strategy = DNNDetectionStrategy(self.config)
            if dnn_strategy.is_available():
                strategies.append(dnn_strategy)
        
        # 전통적 방법 (폴백)
        strategies.append(TraditionalDetectionStrategy(self.config))
        
        return strategies
    
    def _select_strategy(self) -> DetectionStrategy:
        """사용 가능한 첫 번째 전략 선택"""
        for strategy in self.strategies:
            if strategy.is_available():
                printf(f"Using {strategy.__class__.__name__}", ptype=LT.info)
                return strategy
        
        raise RuntimeError("No detection strategy available")
    
    def detect(self, frame: np.ndarray, cam_id: int = 0) -> DetectionResult:
        """볼 검출 실행
        
        Args:
            frame: 입력 프레임
            cam_id: 카메라 ID (로깅용)
            
        Returns:
            DetectionResult: 검출 결과
        """
        with profiling_context(self.profiler):
            result = self.current_strategy.detect(frame)
        
        if result.detection_time > 0:
            self.monitor.record(result.detection_time)
        
        return result
    
    def get_stats(self) -> dict[str, float]:
        """성능 통계"""
        return self.monitor.get_stats()
    
    def save_profile(self, filepath: Path) -> None:
        """프로파일 결과 저장"""
        if not self.profiler:
            printf("Profiling not enabled", ptype=LT.warning)
            return
        
        try:
            # 바이너리 프로파일
            self.profiler.dump_stats(str(filepath))
            
            # 텍스트 리포트
            text_path = filepath.with_suffix('.txt')
            with open(text_path, 'w') as f:
                stats = pstats.Stats(self.profiler, stream=f)
                stats.sort_stats('cumulative')
                stats.print_stats(30)  # 상위 30개만
            
            printf(f"Profile saved to {filepath}", ptype=LT.info)
            
        except IOError as e:
            printf(f"Failed to save profile: {e}", ptype=LT.error)
    
    def switch_strategy(self, strategy_name: str) -> bool:
        """검출 전략 전환
        
        Args:
            strategy_name: 'dnn' 또는 'traditional'
            
        Returns:
            bool: 전환 성공 여부
        """
        strategy_map = {
            'dnn': DNNDetectionStrategy,
            'traditional': TraditionalDetectionStrategy
        }
        
        target_class = strategy_map.get(strategy_name.lower())
        if not target_class:
            return False
        
        for strategy in self.strategies:
            if isinstance(strategy, target_class) and strategy.is_available():
                self.current_strategy = strategy
                printf(f"Switched to {strategy_name}", ptype=LT.info)
                return True
        
        return False
