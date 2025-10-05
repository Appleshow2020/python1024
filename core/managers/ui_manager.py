# core/managers/ui_manager.py
import time
from typing import Dict, List, Optional, Any, Callable
from collections import deque

from core.managers.user_interface import UserInterface
from core.services.animation_service import AnimationService
from utils.printing import printf, LT


class UIStatistics:
    """UI 통계 관리 클래스"""
    
    def __init__(self):
        self.ui_updates = 0
        self.animation_updates = 0
        self.ui_errors = 0
        self.animation_errors = 0
    
    def record_ui_update(self):
        """UI 업데이트 기록"""
        self.ui_updates += 1
    
    def record_animation_update(self):
        """애니메이션 업데이트 기록"""
        self.animation_updates += 1
    
    def record_ui_error(self):
        """UI 에러 기록"""
        self.ui_errors += 1
    
    def record_animation_error(self):
        """애니메이션 에러 기록"""
        self.animation_errors += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'ui_updates': self.ui_updates,
            'animation_updates': self.animation_updates,
            'ui_errors': self.ui_errors,
            'animation_errors': self.animation_errors
        }
    
    def reset(self):
        """통계 초기화"""
        self.ui_updates = 0
        self.animation_updates = 0
        self.ui_errors = 0
        self.animation_errors = 0


class UIManager:
    """UI 컴포넌트 관리자"""
    
    # 클래스 상수
    DEFAULT_UI_UPDATE_INTERVAL = 1.0
    DEFAULT_ANIMATION_UPDATE_INTERVAL = 0.5
    
    # 존 플래그 순서 정의 (확장 가능)
    ZONE_FLAG_ORDER = [
        "On Floor",
        "Hitted",
        "Thrower",
        "OutLined",
        "L In",
        "R In",
        "L Out",
        "R Out",
        "Running"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ui_config = config.get('processing', {}).get('update_intervals', {})
        
        # UI 컴포넌트들
        self.user_interface: Optional[UserInterface] = None
        self.animation_service: Optional[AnimationService] = None
        
        # 업데이트 간격 설정
        self.ui_update_interval = self.ui_config.get(
            'ui', 
            self.DEFAULT_UI_UPDATE_INTERVAL
        )
        self.animation_update_interval = self.ui_config.get(
            'animation', 
            self.DEFAULT_ANIMATION_UPDATE_INTERVAL
        )
        
        # 마지막 업데이트 시간
        self.last_ui_update = 0
        self.last_animation_update = 0
        
        # UI 통계
        self.stats = UIStatistics()
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """UI 컴포넌트 초기화 (내부 헬퍼)"""
        self._initialize_user_interface()
        self._initialize_animation_service()
    
    def _initialize_user_interface(self):
        """UserInterface 초기화 (내부 헬퍼)"""
        try:
            self.user_interface = UserInterface()
            printf("UserInterface initialized", ptype=LT.info)
        except Exception as e:
            printf(f"UserInterface initialization failed: {e}", ptype=LT.warning)
            self.user_interface = None
    
    def _initialize_animation_service(self):
        """AnimationService 초기화 (내부 헬퍼)"""
        try:
            self.animation_service = AnimationService(self.config)
            printf("AnimationService initialized", ptype=LT.info)
        except Exception as e:
            printf(f"AnimationService initialization failed: {e}", ptype=LT.warning)
            self.animation_service = None
    
    def update_referee_ui(self, zone_flags: Dict[str, bool]) -> bool:
        """레퍼리 UI 업데이트"""
        if not self._should_update_ui():
            return False
        
        if not self.user_interface:
            return False
        
        try:
            flag_list = self._convert_zone_flags_to_list(zone_flags)
            self.user_interface.update(flag_list)
            
            self.stats.record_ui_update()
            self.last_ui_update = time.perf_counter()
            return True
            
        except Exception as e:
            printf(f"UI update error: {e}", ptype=LT.warning)
            self.stats.record_ui_error()
            return False
    
    def _should_update_ui(self) -> bool:
        """UI 업데이트 시점 판단 (내부 헬퍼)"""
        current_time = time.perf_counter()
        return current_time - self.last_ui_update >= self.ui_update_interval
    
    def _convert_zone_flags_to_list(self, zone_flags: Dict[str, bool]) -> List[str]:
        """존 플래그를 리스트로 변환 (내부 헬퍼)"""
        return [
            str(zone_flags.get(flag_name, False)).lower()
            for flag_name in self.ZONE_FLAG_ORDER
        ]
    
    def update_animation(self, position_data: Dict[float, Any]) -> bool:
        """애니메이션 업데이트"""
        if not self._should_update_animation():
            return False
        
        if not self.animation_service:
            return False
        
        try:
            success = self.animation_service.update_data(position_data)
            
            if success:
                self.stats.record_animation_update()
                self.last_animation_update = time.perf_counter()
            
            return success
            
        except Exception as e:
            printf(f"Animation update error: {e}", ptype=LT.warning)
            self.stats.record_animation_error()
            return False
    
    def _should_update_animation(self) -> bool:
        """애니메이션 업데이트 시점 판단 (내부 헬퍼)"""
        current_time = time.perf_counter()
        return current_time - self.last_animation_update >= self.animation_update_interval
    
    def process_animation_updates(self) -> bool:
        """애니메이션 업데이트 처리"""
        if not self.animation_service:
            return False
        
        try:
            self.animation_service.process_updates()
            return True
        except Exception as e:
            printf(f"Animation process error: {e}", ptype=LT.warning)
            return False
    
    def force_animation_update(self, position_data: Dict[float, Any]):
        """애니메이션 강제 업데이트"""
        if not self.animation_service:
            return
        
        try:
            self.animation_service.force_update(position_data)
            printf("Animation force updated", ptype=LT.info)
        except Exception as e:
            printf(f"Force animation update error: {e}", ptype=LT.warning)
    
    def get_ui_statistics(self) -> Dict[str, Any]:
        """UI 통계 반환"""
        stats = self.stats.to_dict()
        
        # 애니메이션 성능 통계 추가
        if self.animation_service:
            animation_stats = self.animation_service.get_performance_stats()
            stats.update({
                f"animation_{k}": v 
                for k, v in animation_stats.items()
            })
        
        return stats
    
    def cleanup(self):
        """UI 컴포넌트 정리"""
        printf("Cleaning up UI components...", ptype=LT.info)
        
        self._cleanup_animation_service()
        self._cleanup_matplotlib()
    
    def _cleanup_animation_service(self):
        """AnimationService 정리 (내부 헬퍼)"""
        if not self.animation_service:
            return
        
        try:
            self.animation_service.close()
            printf("AnimationService closed", ptype=LT.info)
        except Exception as e:
            printf(f"AnimationService cleanup error: {e}", ptype=LT.warning)
    
    def _cleanup_matplotlib(self):
        """Matplotlib 정리 (내부 헬퍼)"""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            printf(f"Matplotlib cleanup error: {e}", ptype=LT.warning)