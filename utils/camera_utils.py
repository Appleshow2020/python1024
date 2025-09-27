# utils/camera_utils.py
"""
카메라 관련 유틸리티 함수들
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from classes.printing import printf, LT


def validate_camera_index(device_id: int, timeout: float = 2.0) -> bool:
    """카메라 인덱스 유효성 검사"""
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            return False
        
        # 프레임 읽기 테스트
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None and frame.size > 0
    except Exception:
        return False


def get_camera_properties(device_id: int) -> Dict[str, Any]:
    """카메라 속성 정보 반환"""
    properties = {}
    
    try:
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            properties = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName(),
                'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': cap.get(cv2.CAP_PROP_SATURATION),
                'hue': cap.get(cv2.CAP_PROP_HUE)
            }
        cap.release()
    except Exception as e:
        printf(f"Error getting camera properties for device {device_id}: {e}", ptype=LT.error)
    
    return properties


def optimize_camera_settings(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> bool:
    """카메라 설정 최적화"""
    success_count = 0
    total_settings = 0
    
    settings = [
        (cv2.CAP_PROP_FRAME_WIDTH, width),
        (cv2.CAP_PROP_FRAME_HEIGHT, height),
        (cv2.CAP_PROP_FPS, fps),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_AUTOFOCUS, 0),
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    ]
    
    for prop, value in settings:
        total_settings += 1
        try:
            if cap.set(prop, value):
                success_count += 1
        except Exception:
            pass
    
    return success_count >= (total_settings // 2)


def detect_camera_backends() -> List[str]:
    """사용 가능한 카메라 백엔드 감지"""
    backends = []
    
    # OpenCV에서 지원하는 주요 백엔드들
    backend_names = [
        ('DirectShow', cv2.CAP_DSHOW),
        ('V4L2', cv2.CAP_V4L2), 
        ('AVFoundation', cv2.CAP_AVFOUNDATION),
        ('GStreamer', cv2.CAP_GSTREAMER),
        ('FFMPEG', cv2.CAP_FFMPEG)
    ]
    
    for name, backend_id in backend_names:
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                backends.append(name)
            cap.release()
        except Exception:
            pass
    
    return backends


def calculate_optimal_resolution(available_resolutions: List[Tuple[int, int]], 
                                target_fps: int = 30) -> Tuple[int, int]:
    """최적 해상도 계산"""
    if not available_resolutions:
        return (640, 480)  # 기본값
    
    # 16:9 비율을 선호하고 적당한 해상도 선택
    preferred_resolutions = [
        (1920, 1080), (1280, 720), (960, 540), (640, 360)
    ]
    
    for pref_w, pref_h in preferred_resolutions:
        for avail_w, avail_h in available_resolutions:
            if pref_w == avail_w and pref_h == avail_h:
                return (pref_w, pref_h)
    
    # 적당한 크기의 해상도 선택 (너무 크지도 작지도 않게)
    suitable_resolutions = [
        (w, h) for w, h in available_resolutions 
        if 320 <= w <= 1920 and 240 <= h <= 1080
    ]
    
    if suitable_resolutions:
        # 면적 기준으로 중간 정도 선택
        suitable_resolutions.sort(key=lambda x: x[0] * x[1])
        mid_index = len(suitable_resolutions) // 2
        return suitable_resolutions[mid_index]
    
    return available_resolutions[0] if available_resolutions else (640, 480)