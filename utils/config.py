# utils/config.py
import json
import os
from pathlib import Path
from typing import Dict, Any
from utils.printing import printf, LT
from config.default_config import get_default_config

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    printf("PyYAML not installed. Using JSON config instead.", ptype=LT.warning)


class ConfigManager:
    """설정 파일 관리 클래스"""
    
    def __init__(self):
        self._config = None
        self._config_path = None
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return get_default_config()

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
                printf(f"Configuration loaded from {config_path.name}", ptype=LT.info)
                self._config = {**default_config, **config}
                self._config_path = config_path
                return
            except Exception as e:
                printf(f"Failed to load config: {e}. Using defaults.", ptype=LT.warning)
        
        # 새 설정 파일 생성
        try:
            with open(config_path, 'w') as f:
                if use_yaml:
                    yaml.dump(default_config, f, default_flow_style=False)
                else:
                    json.dump(default_config, f, indent=2)
            printf(f"Default {config_path.name} created", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to create config file: {e}", ptype=LT.warning)
        
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
            printf(f"Configuration saved to {self._config_path.name}", ptype=LT.info)
        except Exception as e:
            printf(f"Failed to save config: {e}", ptype=LT.error)