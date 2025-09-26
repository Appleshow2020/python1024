# core/managers/tracking_manager.py
"""
3D 트래킹 관리자 클래스
기존 BallTracker3D 클래스와 삼각측량 로직을 관리
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

from classes.BallTracker3Dcopy import BallTracker3D
from classes.CameraCalibration import CameraCalibration
from core.services.place_checker import PlaceCheckerService
from core.models.field_zones import FieldZones
from utils.geometry_utils import build_point_grid, make_field_zones
from classes.printing import printf, LT


class TrackingManager:
    """3D 볼 트래킹 및 위치 추적 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_config = config['processing']
        
        # 트래킹 관련 컴포넌트들
        self.tracker: Optional[BallTracker3D] = None
        self.place_checker: Optional[PlaceCheckerService] = None
        self.camera_params: Optional[Dict] = None
        
        # 위치 이력 관리
        history_size = self.processing_config['position_history_size']
        self.position_history = deque(maxlen=history_size)
        
        # 트래킹 통계
        self.tracking_stats = {
            'triangulation_attempts': 0,
            'successful_triangulations': 0,
            'failed_triangulations': 0,
            'position_updates': 0
        }
        
        # 필드 존 설정
        self._initialize_field_zones()
        
    def _initialize_field_zones(self):
        """필드 존 초기화"""
        try:
            point_list = build_point_grid()
            zones = make_field_zones(point_list)
            self.place_checker = PlaceCheckerService(zones)
            printf("Field zones initialized", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to initialize field zones: {e}", ptype=LT.error)
    
    def initialize_calibration(self, selected_cameras: Dict[int, int]) -> bool:
        """카메라 캘리브레이션 초기화"""
        try:
            # 카메라 설정 준비
            cam_configs = []
            for cam_idx in range(1, min(4, len(selected_cameras) + 1)):
                camera_key = str(cam_idx)
                if camera_key in self.config['cameras']:
                    config = {"id": f"cam{cam_idx}"}
                    config.update(self.config['cameras'][camera_key])
                    cam_configs.append(config)
            
            if not cam_configs:
                printf("No camera configurations found", ptype=LT.error)
                return False
            
            # 캘리브레이션 수행
            camera_config = self.config['camera']
            calibrate = CameraCalibration(
                cam_configs, 
                camera_config['width'], 
                camera_config['height'], 
                800, 800  # fx, fy 기본값
            )
            
            self.camera_params = calibrate.get_camera_params()
            printf("Camera calibration completed", ptype=LT.info)
            
            # BallTracker3D 초기화
            self.tracker = BallTracker3D(self.camera_params)
            printf("3D ball tracker initialized", ptype=LT.info)
            
            return True
            
        except Exception as e:
            printf(f"Calibration failed: {e}", ptype=LT.error)
            return False
    
    def process_detections(self, pts_2d: List[Tuple[int, int]], cam_ids: List[int]) -> Optional[Dict[str, Any]]:
        """
        2D 검출 결과를 3D 위치로 변환 및 상태 업데이트
        
        Args:
            pts_2d: 2D 검출 포인트들
            cam_ids: 해당 카메라 ID들
            
        Returns:
            상태 정보 딕셔너리 또는 None
        """
        if not self.tracker:
            printf("Tracker not initialized", ptype=LT.error)
            return None
        
        if len(pts_2d) < 2:
            self.tracking_stats['failed_triangulations'] += 1
            return None
        
        self.tracking_stats['triangulation_attempts'] += 1
        
        try:
            # 3D 위치 계산 (삼각측량)
            position_3d = self.tracker.triangulate_point(pts_2d, cam_ids)
            
            if (position_3d is not None and 
                not np.any(np.isnan(position_3d)) and 
                not np.any(np.isinf(position_3d))):
                
                self.tracking_stats['successful_triangulations'] += 1
                timestamp = time.perf_counter()
                
                # 상태 업데이트 (속도, 방향 계산)
                state = self.tracker.update_state(position_3d, timestamp)
                
                if state.get('position') is not None:
                    # 위치 이력에 추가
                    position_entry = {
                        'timestamp': timestamp,
                        'position': tuple(state['position']),
                        'velocity': tuple(state.get('velocity', (0, 0, 0))),
                        'direction': tuple(state.get('direction', (0, 0, 0))),
                        'confidence': len(pts_2d) / 3.0,  # 최대 3개 카메라 기준
                        'cam_count': len(pts_2d)
                    }
                    
                    self.position_history.append(position_entry)
                    self.tracking_stats['position_updates'] += 1
                    
                    # 볼 위치 확인 (필드 존)
                    zone_info = None
                    if self.place_checker:
                        bx, by = state["position"][0], state["position"][1]
                        zone = self.place_checker.check(bx, by)
                        zone_info = {
                            'zone': zone,
                            'flags': self.place_checker.get_flags()
                        }
                    
                    # 종합 상태 정보 반환
                    return {
                        'position_3d': position_3d,
                        'state': state,
                        'zone_info': zone_info,
                        'position_entry': position_entry,
                        'detection_count': len(pts_2d)
                    }
                    
            else:
                self.tracking_stats['failed_triangulations'] += 1
                
        except Exception as e:
            printf(f"Tracking error: {e}", ptype=LT.error)
            self.tracking_stats['failed_triangulations'] += 1
        
        return None
    
    def get_recent_positions(self, count: int = 10) -> List[Dict[str, Any]]:
        """최근 위치 이력 반환"""
        return list(self.position_history)[-count:]
    
    def get_position_data_for_animation(self) -> Dict[float, Tuple[float, float, float]]:
        """애니메이션용 위치 데이터 반환"""
        return {
            entry['timestamp']: entry['position'] 
            for entry in self.position_history
        }
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """트래킹 통계 반환"""
        stats = self.tracking_stats.copy()
        
        # 성공률 계산
        if stats['triangulation_attempts'] > 0:
            stats['success_rate'] = (
                stats['successful_triangulations'] / 
                stats['triangulation_attempts'] * 100
            )
        else:
            stats['success_rate'] = 0.0
        
        # 추가 정보
        stats['position_history_size'] = len(self.position_history)
        stats['latest_position'] = None
        
        if self.position_history:
            latest = self.position_history[-1]
            stats['latest_position'] = {
                'position': latest['position'],
                'timestamp': latest['timestamp'],
                'confidence': latest['confidence']
            }
        
        return stats
    
    def get_current_zone_flags(self) -> Optional[Dict[str, bool]]:
        """현재 필드 존 플래그들 반환"""
        if self.place_checker:
            return self.place_checker.get_flags()
        return None
    
    def reset_tracking_data(self):
        """트래킹 데이터 및 통계 초기화"""
        self.position_history.clear()
        self.tracking_stats = {
            'triangulation_attempts': 0,
            'successful_triangulations': 0,
            'failed_triangulations': 0,
            'position_updates': 0
        }
        
        if self.tracker:
            # BallTracker3D의 이전 위치 데이터도 초기화
            self.tracker.prev_positions.clear()
            self.tracker.prev_times.clear()
        
        printf("Tracking data reset", ptype=LT.info)