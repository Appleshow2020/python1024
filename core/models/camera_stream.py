# core/models/camera_stream.py
"""
카메라 스트림 데이터 모델
기존 CamStream 데이터클래스를 별도 모듈로 분리
"""

import cv2
from dataclasses import dataclass, field
from collections import deque
from threading import Lock
from typing import Optional, Dict, Any, Deque, Tuple
import numpy as np


@dataclass
class CamStream:
    """카메라 스트림 데이터 모델"""
    cap: Optional[cv2.VideoCapture]
    frames: Deque = field(default_factory=lambda: deque(maxlen=3))
    last_detection: Optional[Tuple[float, float]] = None
    last_frame_time: float = 0.0
    detection_cache: Optional[Tuple[float, float]] = None
    cache_time: float = 0.0
    consecutive_failures: int = 0
    lock: Lock = field(default_factory=Lock)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """카메라가 활성 상태인지 확인"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_latest_frame(self):
        """최신 프레임 반환 (스레드 안전)"""
        with self.lock:
            if self.frames:
                return self.frames[-1].copy()
        return None
    
    def has_recent_detection(self, timeout: float = 1.0) -> bool:
        """최근 검출 결과가 있는지 확인"""
        import time
        if self.detection_cache is None:
            return False
        return (time.perf_counter() - self.cache_time) < timeout