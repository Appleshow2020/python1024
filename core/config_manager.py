# core/config_manager.py
import json
import os

class ConfigManager:
    """JSON 설정 파일을 읽고 관리하는 클래스"""

    def __init__(self, config_path='config.json'):
        """
        ConfigManager를 초기화합니다.
        :param config_path: 설정 파일의 경로
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """설정 파일에서 설정을 로드합니다."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"'{self.config_path}' 파일 파싱 중 오류 발생: {e}")
            return None
        except Exception as e:
            print(f"설정 파일 로드 중 예외 발생: {e}")
            return None

    def get_config(self):
        """로드된 설정 객체를 반환합니다."""
        return self.config

    def get_section(self, section_name):
        """설정의 특정 섹션을 반환합니다."""
        return self.config.get(section_name, {})