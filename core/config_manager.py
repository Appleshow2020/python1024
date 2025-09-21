# core/config_manager.py
import json
import os
from classes.printing import *
import yaml
from pathlib import Path


class ConfigManager:
    """Class for reading and managing JSON configuration files."""

    def __init__(self, config_path='config.json'):
        """
        Initialize ConfigManager.
        :param config_path: Path to the configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            import yaml
            YAML_AVAILABLE = True
        except ImportError:
            YAML_AVAILABLE = False
            printf("PyYAML not installed. Using JSON config instead.", LT.warning)
        """설정 파일 로드 또는 기본값 생성 - JSON 우선, YAML 선택적"""
        default_config = {
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
                return {**default_config, **config}
            except Exception as e:
                printf(f"Failed to load config: {e}. Using defaults.", LT.warning)
    
    def get_config(self):
        """Return the loaded configuration object."""
        return self.config

    def get_section(self, section_name):
        """Return a specific section of the configuration."""
        return self.config.get(section_name, {})
    
