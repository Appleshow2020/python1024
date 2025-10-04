# core/managers/detection_manager.py
import cv2
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from core.services.ball_detector import BallDetectorService
from core.managers.camera_manager import CameraManager
from utils.printing import printf, LT

class DetectionStatistics:
    """검출 통계 관리 클래스"""
    
    DETECTION_TIMES_MAXLEN = 100
    
    def __init__(self):
        self.total_detections = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.detection_times = deque(maxlen=self.DETECTION_TIMES_MAXLEN)
    
    def record_success(self, detection_time: float):
        """성공한 검출 기록"""
        self.successful_detections += 1
        self.detection_times.append(detection_time)
    
    def record_failure(self, detection_time: float):
        """실패한 검출 기록"""
        self.failed_detections += 1
        self.detection_times.append(detection_time)
    
    def add_total(self, count: int):
        """총 검출 시도 횟수 추가"""
        self.total_detections += count
    
    def get_success_rate(self) -> float:
        """성공률 계산"""
        if self.total_detections == 0:
            return 0.0
        return (self.successful_detections / self.total_detections) * 100
    
    def get_avg_detection_time(self) -> float:
        """평균 검출 시간 계산"""
        if not self.detection_times:
            return 0.0
        return np.mean(self.detection_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'failed_detections': self.failed_detections,
            'success_rate': self.get_success_rate(),
            'avg_detection_time': self.get_avg_detection_time()
        }
    
    def reset(self):
        """통계 초기화"""
        self.total_detections = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.detection_times.clear()


class DetectionManager:
    """
    DetectionManager manages the detection process for multiple camera streams,
    tracks detection statistics, and provides visual feedback and display.
    """
    
    # 클래스 상수
    STATS_UPDATE_INTERVAL_FRAMES = 30
    
    def __init__(self, camera_manager: CameraManager, config: Dict[str, Any]):
        self.camera_manager = camera_manager
        self.config = config
        self.detection_config = config['detection']
        self.display_config = config['display']
        
        # 검출 서비스 초기화
        self.detector = BallDetectorService(self.detection_config)
        
        # 통계 객체
        self.stats = DetectionStatistics()
        
        # 프레임 카운터
        self.frame_count = 0
        
        # 디스플레이 설정 캐싱
        self._cache_display_settings()
    
    def _cache_display_settings(self):
        """디스플레이 설정 캐싱 (내부 헬퍼)"""
        self.circle_radius = self.display_config['circle_radius']
        self.line_thickness = self.display_config['line_thickness']
    
    def process_frame_detections(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """프레임 검출 처리"""
        self.frame_count += 1
        
        snapshot = self.camera_manager.get_frame_snapshot()
        if not snapshot:
            return [], []
        
        pts_2d, cam_ids = self._process_camera_frames(snapshot)
        
        self.stats.add_total(len(snapshot))
        return pts_2d, cam_ids
    
    def _process_camera_frames(
        self, 
        snapshot: Dict[int, np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """모든 카메라 프레임 처리 (내부 헬퍼)"""
        pts_2d = []
        cam_ids = []
        
        for cam_id, frame in snapshot.items():
            detection_result = self._process_single_camera(cam_id, frame)
            
            if detection_result is not None:
                pts_2d.append(detection_result)
                cam_ids.append(cam_id)
        
        return pts_2d, cam_ids
    
    def _process_single_camera(
        self, 
        cam_id: int, 
        frame: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """단일 카메라 프레임 처리 (내부 헬퍼)"""
        detection_start = time.perf_counter()
        
        try:
            pt, has_detection = self.detector.detect(frame, cam_id, self.frame_count)
            detection_time = time.perf_counter() - detection_start
            
            if pt is not None and has_detection:
                self._handle_successful_detection(cam_id, frame, pt, detection_time)
                return pt
            else:
                self.stats.record_failure(detection_time)
                
        except Exception as e:
            detection_time = time.perf_counter() - detection_start
            printf(f"Detection error cam{cam_id}: {e}", ptype=LT.error)
            self.stats.record_failure(detection_time)
        
        finally:
            self._display_frame(frame, cam_id)
        
        return None
    
    def _handle_successful_detection(
        self,
        cam_id: int,
        frame: np.ndarray,
        pt: Tuple[int, int],
        detection_time: float
    ):
        """성공한 검출 처리 (내부 헬퍼)"""
        self.stats.record_success(detection_time)
        self._update_camera_detection_cache(cam_id, pt)
        self._add_visual_feedback(frame, pt, cam_id)
    
    def _update_camera_detection_cache(self, cam_id: int, pt: Tuple[int, int]):
        """카메라 검출 캐시 업데이트 (내부 헬퍼)"""
        stream = self.camera_manager.streams.get(cam_id)
        if stream:
            stream.detection_cache = pt
            stream.cache_time = time.perf_counter()
    
    def _add_visual_feedback(
        self, 
        frame: np.ndarray, 
        pt: Tuple[int, int], 
        cam_id: int
    ):
        """시각적 피드백 추가"""
        # 검출된 볼 위치에 원 그리기
        pt_int = tuple(map(int, pt))
        cv2.circle(frame, pt_int, self.circle_radius, (0, 255, 0), self.line_thickness)
        
        # 검출 FPS 표시
        self._draw_detection_fps(frame)
    
    def _draw_detection_fps(self, frame: np.ndarray):
        """검출 FPS 표시 (내부 헬퍼)"""
        detection_stats = self.detector.get_stats()
        if not detection_stats or 'detection_fps' not in detection_stats:
            return
        
        info_text = f"Det FPS: {detection_stats['detection_fps']:.1f}"
        cv2.putText(
            frame, 
            info_text, 
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 255, 255), 
            1
        )
    
    def _display_frame(self, frame: np.ndarray, cam_id: int):
        """프레임 표시"""
        # 주기적으로 카메라 통계 표시
        if self._should_update_stats_display():
            self._draw_camera_stats(frame, cam_id)
        
        cv2.imshow(f"CAM{cam_id}", frame)
    
    def _should_update_stats_display(self) -> bool:
        """통계 표시 업데이트 시점 판단 (내부 헬퍼)"""
        return self.frame_count % self.STATS_UPDATE_INTERVAL_FRAMES == 0
    
    def _draw_camera_stats(self, frame: np.ndarray, cam_id: int):
        """카메라 통계 표시 (내부 헬퍼)"""
        stream = self.camera_manager.streams.get(cam_id)
        if not stream:
            return
        
        cam_stats = stream.stats
        info = (
            f"FPS:{cam_stats.get('avg_fps', 0):.1f} "
            f"D:{self.stats.successful_detections} "
            f"F:{self.frame_count}"
        )
        
        cv2.putText(
            frame, 
            info, 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1
        )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """검출 통계 반환"""
        base_stats = self.stats.to_dict()
        
        # 검출기 통계 추가
        detector_stats = self.detector.get_stats()
        if detector_stats:
            base_stats.update(detector_stats)
        
        return base_stats
    
    def reset_statistics(self):
        """통계 초기화"""
        self.stats.reset()
        self.frame_count = 0
        printf("Detection statistics reset", ptype=LT.info)
    
    def save_detection_profile(self, filename: str = "detection_profile.prof"):
        """검출 프로파일 저장"""
        self.detector.save_profile(filename)