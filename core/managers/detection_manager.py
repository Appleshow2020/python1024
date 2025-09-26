# core/managers/detection_manager.py
"""
볼 검출 관리자 클래스
기존 ProfiledBallDetector 클래스를 통합하여 검출 프로세스를 관리
"""

import cv2
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from core.services.ball_detector import BallDetectorService
from core.managers.camera_manager import CameraManager
from classes.printing import printf, LT


class DetectionManager:
    """볼 검출 관리자 클래스"""
    
    def __init__(self, camera_manager: CameraManager, config: Dict[str, Any]):
        self.camera_manager = camera_manager
        self.config = config
        self.detection_config = config['detection']
        self.display_config = config['display']
        
        # 검출 서비스 초기화
        self.detector = BallDetectorService(self.detection_config)
        
        # 검출 통계
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'detection_times': deque(maxlen=100)
        }
        
        # 프레임 카운터
        self.frame_count = 0
        
    def process_frame_detections(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        현재 프레임에서 모든 카메라의 볼 검출 수행
        Returns:
            pts_2d: 2D 검출 포인트들
            cam_ids: 해당 카메라 ID들
        """
        self.frame_count += 1
        pts_2d = []
        cam_ids = []
        
        # 모든 활성 카메라에서 프레임 수집
        snapshot = self.camera_manager.get_frame_snapshot()
        
        if not snapshot:
            return pts_2d, cam_ids
        
        # 각 카메라에서 볼 검출
        for cam_id, frame in snapshot.items():
            detection_start = time.perf_counter()
            
            try:
                # 볼 검출 수행
                pt, has_detection = self.detector.detect(frame, cam_id, self.frame_count)
                
                if pt is not None and has_detection:
                    pts_2d.append(pt)
                    cam_ids.append(cam_id)
                    
                    self.detection_stats['successful_detections'] += 1
                    
                    # 카메라 스트림의 검출 캐시 업데이트
                    stream = self.camera_manager.streams.get(cam_id)
                    if stream:
                        stream.detection_cache = pt
                        stream.cache_time = time.perf_counter()
                    
                    # 시각적 피드백 추가
                    self._add_visual_feedback(frame, pt, cam_id)
                
                else:
                    self.detection_stats['failed_detections'] += 1
                
                # 검출 시간 기록
                detection_time = time.perf_counter() - detection_start
                self.detection_stats['detection_times'].append(detection_time)
                
                # 프레임 표시
                self._display_frame(frame, cam_id)
                
            except Exception as e:
                printf(f"Detection error cam{cam_id}: {e}", LT.error)
                self.detection_stats['failed_detections'] += 1
        
        self.detection_stats['total_detections'] += len(snapshot)
        return pts_2d, cam_ids
    
    def _add_visual_feedback(self, frame: np.ndarray, pt: Tuple[int, int], cam_id: int):
        """검출된 볼에 시각적 피드백 추가"""
        radius = self.display_config['circle_radius']
        thickness = self.display_config['line_thickness']
        
        # 검출된 볼 위치에 원 그리기
        cv2.circle(frame, tuple(map(int, pt)), radius, (0, 255, 0), thickness)
        
        # 검출 정확도 표시
        detection_stats = self.detector.get_stats()
        if detection_stats and 'detection_fps' in detection_stats:
            info_text = f"Det FPS: {detection_stats['detection_fps']:.1f}"
            cv2.putText(frame, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def _display_frame(self, frame: np.ndarray, cam_id: int):
        """프레임 표시 및 상태 정보 추가"""
        # 카메라 통계 표시 (주기적으로)
        if self.frame_count % 30 == 0:
            cam_stats = self.camera_manager.streams[cam_id].stats
            info = (f"FPS:{cam_stats.get('avg_fps', 0):.1f} "
                   f"D:{self.detection_stats['successful_detections']} "
                   f"F:{self.frame_count}")
            cv2.putText(frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 프레임 표시
        cv2.imshow(f"CAM{cam_id}", frame)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """검출 통계 반환"""
        base_stats = {
            'total_detections': self.detection_stats['total_detections'],
            'successful_detections': self.detection_stats['successful_detections'],
            'failed_detections': self.detection_stats['failed_detections'],
            'success_rate': 0.0,
            'avg_detection_time': 0.0
        }
        
        if self.detection_stats['total_detections'] > 0:
            base_stats['success_rate'] = (
                self.detection_stats['successful_detections'] / 
                self.detection_stats['total_detections'] * 100
            )
        
        if self.detection_stats['detection_times']:
            times = list(self.detection_stats['detection_times'])
            base_stats['avg_detection_time'] = np.mean(times)
        
        # 검출기 통계 추가
        detector_stats = self.detector.get_stats()
        if detector_stats:
            base_stats.update(detector_stats)
        
        return base_stats
    
    def reset_statistics(self):
        """검출 통계 초기화"""
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'detection_times': deque(maxlen=100)
        }
        self.frame_count = 0
        printf("Detection statistics reset", LT.info)
    
    def save_detection_profile(self, filename: str = "detection_profile.prof"):
        """검출기 프로파일 저장"""
        self.detector.save_profile(filename)