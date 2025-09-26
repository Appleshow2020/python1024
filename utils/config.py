# utils/config.py
"""
설정 관리 모듈
기존의 load_config 함수를 클래스 기반으로 리팩터링
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from classes.printing import printf, LT

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    printf("PyYAML not installed. Using JSON config instead.", LT.warning)


class ConfigManager:
    """설정 파일 관리 클래스"""
    
    def __init__(self):
        self._config = None
        self._config_path = None
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정값 반환"""
        return {
            'camera': {
                'width': 640,
                'height': 360,
                'fps': 30,
                'detection_interval': 3,
                'buffer_size': 3,
                'search_range': 10
            },
            'processing': {
                'update_intervals': {
                    'ui': 1.0,
                    'animation': 0.5,
                    'stats': 5.0
                },
                'position_history_size': 100,
                'queue_size': 10
            },
            'detection': {
                'hsv_lower': [0, 50, 50],
                'hsv_upper': [15, 255, 255],
                'min_contour_area': 30,
                'morphology_kernel_size': 3,
                'enable_gpu': False,
                'model_path': 'ball_detection.onnx'
            },
            'display': {
                'plot_size': [8, 6],
                'max_plot_points': 10,
                'circle_radius': 5,
                'line_thickness': 2
            },
            'cameras': {
                '1': {"position": [-0.47, -0.52, 0.19], "rotation": [-30, 45, -10]},
                '2': {"position": [0.05, 0.05, 0.62], "rotation": [-90, 0, 100]},
                '3': {"position": [0.61, 0.39, 0.19], "rotation": [-20, -120, 0]}
            },
            'logging': {
                'level': 'INFO',
                'file': 'ball_tracker.log',
                'max_file_size': '10MB'
            },
            'profiling': {
                'enabled': False,
                'save_interval': 100,
                'output_dir': 'profiles'
            }
        }
    
    def _load_config(self):
        """설정 파일 로드"""
        default_config = self._get_default_config()
        
        # JSON 파일 우선 확인
        json_config_path = Path("config.json")
        yaml_config_path = Path("config.yaml")
        
        config_path = None
        use_yaml = False
        
        if json_config_path.exists():
            config_path = json_config_path
            use_yaml = False
        elif yaml_config_path.exists() and YAML_AVAILABLE:
            config_path = yaml_config_path
            use_yaml = True
        else:
            # 새 파일 생성
            if YAML_AVAILABLE:
                config_path = yaml_config_path
                use_yaml = True
            else:
                config_path = json_config_path
                use_yaml = False
        
        # 기존 파일 로드
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if use_yaml:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                printf(f"Configuration loaded from {config_path.name}", LT.info)
                self._config = {**default_config, **config}
                self._config_path = config_path
                return
            except Exception as e:
                printf(f"Failed to load config: {e}. Using defaults.", LT.warning)
        
        # 새 설정 파일 생성
        try:
            with open(config_path, 'w') as f:
                if use_yaml:
                    yaml.dump(default_config, f, default_flow_style=False)
                else:
                    json.dump(default_config, f, indent=2)
            printf(f"Default {config_path.name} created", LT.info)
        except Exception as e:
            printf(f"Failed to create config file: {e}", LT.warning)
        
        self._config = default_config
        self._config_path = config_path
    
    def get_config(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return self._config
    
    def get(self, key: str, default=None):
        """특정 키의 설정값 반환"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def save_config(self):
        """현재 설정을 파일에 저장"""
        if not self._config_path:
            return
        
        try:
            use_yaml = self._config_path.suffix == '.yaml'
            with open(self._config_path, 'w') as f:
                if use_yaml and YAML_AVAILABLE:
                    yaml.dump(self._config, f, default_flow_style=False)
                else:
                    json.dump(self._config, f, indent=2)
            printf(f"Configuration saved to {self._config_path.name}", LT.info)
        except Exception as e:
            printf(f"Failed to save config: {e}", LT.error)