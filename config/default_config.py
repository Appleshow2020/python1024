# config/default_config.py
from typing import Dict, Any
import json

def get_default_config() -> Dict[str, Any]:
    """기본 설정값 반환"""
    with open("config/config.json", "r") as f:
        return json.load(f)


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