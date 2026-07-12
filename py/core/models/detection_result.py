# core/models/detection_result.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import time


@dataclass
class DetectionResult:
    """볼 검출 결과 데이터 모델"""
    position_2d: Optional[Tuple[int, int]]
    confidence: float
    timestamp: float
    camera_id: int
    detection_time_ms: float
    has_detection: bool
    
    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.perf_counter()
    
    @classmethod
    def create_empty(cls, camera_id: int) -> 'DetectionResult':
        """빈 검출 결과 생성"""
        return cls(
            position_2d=None,
            confidence=0.0,
            timestamp=time.perf_counter(),
            camera_id=camera_id,
            detection_time_ms=0.0,
            has_detection=False
        )
    
    @classmethod
    def create_successful(cls, position_2d: Tuple[int, int], camera_id: int, 
                         detection_time_ms: float, confidence: float = 1.0) -> 'DetectionResult':
        """성공한 검출 결과 생성"""
        return cls(
            position_2d=position_2d,
            confidence=confidence,
            timestamp=time.perf_counter(),
            camera_id=camera_id,
            detection_time_ms=detection_time_ms,
            has_detection=True
        )
    
    def is_valid(self) -> bool:
        """검출 결과 유효성 확인"""
        return (self.has_detection and 
                self.position_2d is not None and 
                self.confidence > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'position_2d': self.position_2d,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'detection_time_ms': self.detection_time_ms,
            'has_detection': self.has_detection
        }


@dataclass 
class MultiCameraDetectionResult:
    """다중 카메라 검출 결과"""
    detections: Dict[int, DetectionResult]
    timestamp: float
    total_detections: int
    successful_detections: int
    
    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.perf_counter()
        
        # 통계 자동 계산
        self.total_detections = len(self.detections)
        self.successful_detections = sum(1 for d in self.detections.values() if d.has_detection)
    
    def get_valid_detections(self) -> Dict[int, DetectionResult]:
        """유효한 검출 결과만 반환"""
        return {
            cam_id: result 
            for cam_id, result in self.detections.items() 
            if result.is_valid()
        }
    
    def get_2d_points_and_cameras(self) -> Tuple[list, list]:
        """2D 포인트와 카메라 ID 리스트 반환"""
        valid_detections = self.get_valid_detections()
        
        pts_2d = [result.position_2d for result in valid_detections.values()]
        cam_ids = list(valid_detections.keys())
        
        return pts_2d, cam_ids
    
    def has_enough_for_triangulation(self, min_cameras: int = 2) -> bool:
        """삼각측량에 충분한 검출 결과가 있는지 확인"""
        return self.successful_detections >= min_cameras