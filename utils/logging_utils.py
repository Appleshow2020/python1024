# utils/logging_utils.py
"""
로깅 관리 유틸리티
기존 setup_logging 함수를 클래스 기반으로 리팩터링
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any
from classes.printing import printf, LT


class LoggingManager:
    """간단하고 안정적인 로깅 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_config = config.get('logging', {})
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        try:
            # 로그 디렉토리 생성
            log_file = Path(self.log_config.get('file', 'ball_tracker.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 기본 로깅 설정
            log_level = getattr(logging, self.log_config.get('level', 'INFO'), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(levelname)s - %(message)s',
                force=True  # 기존 설정 덮어쓰기
            )
            
            self.logger = logging.getLogger('BallTracker')
            self.logger.setLevel(log_level)
            
            # 기존 핸들러 제거
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            
            # 파일 핸들러 추가
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                self.logger.addHandler(file_handler)
            except Exception as e:
                printf(f"File logging failed: {e}. Using console only.", ptype=LT.warning)
            
            # 콘솔 핸들러 추가
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)
            
            printf("Logging system initialized", ptype=LT.info)
            self.logger.info("Ball Tracker logging started")
            
        except Exception as e:
            printf(f"Logging setup failed: {e}. Using dummy logger.", ptype=LT.warning)
            self.logger = self._create_dummy_logger()
    
    def _create_dummy_logger(self):
        """더미 로거 생성 (최후의 수단)"""
        class DummyLogger:
            def info(self, msg): printf(f"INFO: {msg}", ptype=LT.info)
            def warning(self, msg): printf(f"WARNING: {msg}", ptype=LT.warning)
            def error(self, msg): printf(f"ERROR: {msg}", ptype=LT.error)
            def debug(self, msg): printf(f"DEBUG: {msg}", ptype=LT.debug)
        
        return DummyLogger()
    
    def get_logger(self):
        """로거 인스턴스 반환"""
        return self.logger