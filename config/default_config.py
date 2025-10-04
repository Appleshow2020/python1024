# config/default_config.py
from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """기본 설정값 반환"""
    return {
        'camera': {
            'width': 640,
            'height': 360,
            'fps': 30,
            'detection_interval': 3,
            'buffer_size': 3,
            'search_range': 20,
            'auto_exposure': 0.25,
            'enable_autofocus': False
        },
        'processing': {
            'update_intervals': {
                'ui': 1.0,
                'animation': 0.5,
                'stats': 5.0
            },
            'position_history_size': 100,
            'queue_size': 10,
            'frame_skip_threshold': 2.0,
            'max_processing_time_ms': 50.0
        },
        'detection': {
            'hsv_lower': [0, 50, 50],
            'hsv_upper': [15, 255, 255],
            'min_contour_area': 30,
            'max_contour_area': 5000,
            'morphology_kernel_size': 3,
            'enable_gpu': False,
            'model_path': 'ball_detection.onnx',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'profiling_enabled': False
        },
        'display': {
            'plot_size': [8, 6],
            'max_plot_points': 10,
            'circle_radius': 5,
            'line_thickness': 2,
            'show_debug_info': True,
            'enable_blitting': True,
            'animation_interval_ms': 100
        },
        'cameras': {
            '1': {
                "position": [-0.47, -0.52, 0.19], 
                "rotation": [-30, 45, -10]
            },
            '2': {
                "position": [0.05, 0.05, 0.62], 
                "rotation": [-90, 0, 100]
            },
            '3': {
                "position": [0.61, 0.39, 0.19], 
                "rotation": [-20, -120, 0]
            }
        },
        'calibration': {
            'intrinsic_fx': 800,
            'intrinsic_fy': 800,
            'method': 'manual',  # 'manual' or 'automatic'
            'checkerboard_size': (9, 6),
            'square_size': 25.0  # mm
        },
        'tracking': {
            'min_cameras_for_triangulation': 2,
            'max_triangulation_error': 10.0,
            'velocity_smoothing': 0.8,
            'position_smoothing': 0.5,
            'outlier_threshold': 5.0
        },
        'field': {
            'grid_points': {
                'x': [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11],
                'y': [7, 7, 7, 7, 4, 4, 4, 4, -4, -4, -4, -4, -7, -7, -7, -7]
            },
            'zones': {
                'left_in': {'p1_idx': 4, 'p2_idx': 0},
                'right_in': {'p1_idx': 7, 'p2_idx': 3},
                'left_out': {'p1_idx': 12, 'p2_idx': 15},
                'right_out': {'p1_idx': 8, 'p2_idx': 11}
            }
        },
        'logging': {
            'level': 'INFO',
            'file': 'logs/ball_tracker.log',
            'max_file_size': '10MB',
            'backup_count': 5,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console_level': 'INFO'
        },
        'profiling': {
            'enabled': False,
            'save_interval': 100,
            'output_dir': 'profiles',
            'profile_detection': False,
            'profile_tracking': False,
            'profile_ui': False
        },
        'performance': {
            'enable_monitoring': True,
            'stats_interval': 3.0,
            'frame_time_history_size': 1000,
            'system_stats_history_size': 100,
            'fps_threshold': 15.0,
            'cpu_threshold': 80.0,
            'memory_threshold_mb': 1000.0
        },
        'ui': {
            'enable_referee_ui': True,
            'enable_animation': True,
            'enable_performance_dashboard': False,
            'window_positions': {
                'camera_windows': 'auto',
                'referee_ui': (100, 100),
                'animation': (800, 100)
            }
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """설정값 유효성 검사"""
    try:
        # 필수 섹션 확인
        required_sections = ['camera', 'processing', 'detection', 'cameras']
        for section in required_sections:
            if section not in config:
                return False
        
        # 카메라 설정 확인
        camera_config = config['camera']
        if (camera_config.get('width', 0) <= 0 or 
            camera_config.get('height', 0) <= 0 or
            camera_config.get('fps', 0) <= 0):
            return False
        
        # 검출 설정 확인
        detection_config = config['detection']
        hsv_lower = detection_config.get('hsv_lower', [])
        hsv_upper = detection_config.get('hsv_upper', [])
        if len(hsv_lower) != 3 or len(hsv_upper) != 3:
            return False
        
        return True
        
    except Exception:
        return False


def merge_configs(base_config: Dict[str, Any] | None = get_default_config(), user_config: Dict[str, Any]|None = Exception()) -> Dict[str, Any]:
    """설정 병합 (재귀적으로 딕셔너리 병합)"""
    result = base_config.copy()
    
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result