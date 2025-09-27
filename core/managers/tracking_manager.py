# core/managers/tracking_manager.py
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
    """
    TrackingManager is responsible for managing the 3D ball tracking process, including calibration, detection processing, position history, and field zone checks.
    Attributes:
        config (Dict[str, Any]): Configuration dictionary for tracking and camera settings.
        processing_config (Dict): Subset of config related to processing parameters.
        tracker (Optional[BallTracker3D]): 3D ball tracker instance for triangulation and state updates.
        place_checker (Optional[PlaceCheckerService]): Service for checking ball position within field zones.
        camera_params (Optional[Dict]): Camera calibration parameters.
        position_history (deque): History of tracked positions with a fixed maximum length.
        tracking_stats (dict): Statistics on tracking performance and triangulation attempts.
    Methods:
        __init__(config):
            Initializes the TrackingManager with configuration and sets up field zones and history.
        _initialize_field_zones():
            Initializes field zones and the place checker service.
        initialize_calibration(selected_cameras: Dict[int, int]) -> bool:
            Performs camera calibration using selected cameras and initializes the 3D tracker.
        process_detections(pts_2d: List[Tuple[int, int]], cam_ids: List[int]) -> Optional[Dict[str, Any]]:
            Processes 2D detections from multiple cameras, performs triangulation, updates state, and manages position history.
        get_recent_positions(count: int = 10) -> List[Dict[str, Any]]:
            Returns the most recent tracked positions up to the specified count.
        get_position_data_for_animation() -> Dict[float, Tuple[float, float, float]]:
            Returns a mapping of timestamps to 3D positions for animation purposes.
        get_tracking_statistics() -> Dict[str, Any]:
            Returns current tracking statistics, including success rate and latest position.
        get_current_zone_flags() -> Optional[Dict[str, bool]]:
            Returns the current field zone flags from the place checker.
        reset_tracking_data():
            Clears position history and resets tracking statistics and tracker state.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_config = config['processing']
        
        # Tracking related components
        self.tracker: Optional[BallTracker3D] = None
        self.place_checker: Optional[PlaceCheckerService] = None
        self.camera_params: Optional[Dict] = None
        
        # Position history management
        history_size = self.processing_config['position_history_size']
        self.position_history = deque(maxlen=history_size)
        
        # Tracking statistics
        self.tracking_stats = {
            'triangulation_attempts': 0,
            'successful_triangulations': 0,
            'failed_triangulations': 0,
            'position_updates': 0
        }
        
        # Field zone setup
        self._initialize_field_zones()
        
    def _initialize_field_zones(self):
        try:
            point_list = build_point_grid()
            zones = make_field_zones(point_list)
            self.place_checker = PlaceCheckerService(zones)
            printf("Field zones initialized", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to initialize field zones: {e}", ptype=LT.error)
    
    def initialize_calibration(self, selected_cameras: Dict[int, int]) -> bool:
        try:
            # Prepare camera settings
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
            
            # Perform calibration
            camera_config = self.config['camera']
            calibrate = CameraCalibration(
                cam_configs, 
                camera_config['width'], 
                camera_config['height'], 
                800, 800  # fx, fy default values
            )
            
            self.camera_params = calibrate.get_camera_params()
            printf("Camera calibration completed", ptype=LT.info)
            
            # Initialize BallTracker3D
            self.tracker = BallTracker3D(self.camera_params)
            printf("3D ball tracker initialized", ptype=LT.info)
            
            return True
            
        except Exception as e:
            printf(f"Calibration failed: {e}", ptype=LT.error)
            return False
    
    def process_detections(self, pts_2d: List[Tuple[int, int]], cam_ids: List[int]) -> Optional[Dict[str, Any]]:
        if not self.tracker:
            printf("Tracker not initialized", ptype=LT.error)
            return None
        
        if len(pts_2d) < 2:
            self.tracking_stats['failed_triangulations'] += 1
            return None
        
        self.tracking_stats['triangulation_attempts'] += 1
        
        try:
            # Calculate 3D position (triangulation)
            position_3d = self.tracker.triangulate_point(pts_2d, cam_ids)
            
            if (position_3d is not None and 
                not np.any(np.isnan(position_3d)) and 
                not np.any(np.isinf(position_3d))):
                
                self.tracking_stats['successful_triangulations'] += 1
                timestamp = time.perf_counter()
                
                # Update state (calculate velocity, direction)
                state = self.tracker.update_state(position_3d, timestamp)
                
                if state.get('position') is not None:
                    # Add to position history
                    position_entry = {
                        'timestamp': timestamp,
                        'position': tuple(state['position']),
                        'velocity': tuple(state.get('velocity', (0, 0, 0))),
                        'direction': tuple(state.get('direction', (0, 0, 0))),
                        'confidence': len(pts_2d) / 3.0,  # Based on max 3 cameras
                        'cam_count': len(pts_2d)
                    }
                    
                    self.position_history.append(position_entry)
                    self.tracking_stats['position_updates'] += 1
                    
                    # Check ball position (field zone)
                    zone_info = None
                    if self.place_checker:
                        bx, by = state["position"][0], state["position"][1]
                        zone = self.place_checker.check(bx, by)
                        zone_info = {
                            'zone': zone,
                            'flags': self.place_checker.get_flags()
                        }
                    
                    # Return comprehensive state info
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
        return list(self.position_history)[-count:]
    
    def get_position_data_for_animation(self) -> Dict[float, Tuple[float, float, float]]:
        return {
            entry['timestamp']: entry['position'] 
            for entry in self.position_history
        }
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        stats = self.tracking_stats.copy()
        
        # Calculate success rate
        if stats['triangulation_attempts'] > 0:
            stats['success_rate'] = (
                stats['successful_triangulations'] / 
                stats['triangulation_attempts'] * 100
            )
        else:
            stats['success_rate'] = 0.0
        
        # Additional info
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
        if self.place_checker:
            return self.place_checker.get_flags()
        return None
    
    def reset_tracking_data(self):
        self.position_history.clear()
        self.tracking_stats = {
            'triangulation_attempts': 0,
            'successful_triangulations': 0,
            'failed_triangulations': 0,
            'position_updates': 0
        }
        
        if self.tracker:
            # Also clear previous position data in BallTracker3D
            self.tracker.prev_positions.clear()
            self.tracker.prev_times.clear()
        
        printf("Tracking data reset", ptype=LT.info)