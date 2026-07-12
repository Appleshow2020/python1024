# config/default_config.py
from typing import Dict, Any
import json


def get_default_config() -> Dict[str, Any]:
    """기본 설정값 반환"""
    with open("config.json", "r", encoding="utf-8") as f:
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


def merge_configs(base_config: Dict[str, Any] | None = None,
                   user_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """설정 병합 (재귀적으로 딕셔너리 병합)"""
    if base_config is None:
        base_config = get_default_config()
    if user_config is None:
        user_config = {}

    result = base_config.copy()

    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class ConfigManager:
    """sensor_controller 등에서 정적 접근용으로 사용하는 래퍼 클래스"""

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return get_default_config()

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        return validate_config(config)

    @staticmethod
    def merge_configs(base_config: Dict[str, Any] | None = None,
                       user_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return merge_configs(base_config, user_config)