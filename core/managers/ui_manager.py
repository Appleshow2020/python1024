# core/managers/ui_manager.py
import time
from typing import Dict, List, Optional, Any
from collections import deque

from core.managers.user_interface import UserInterface
from core.services.animation_service import AnimationService
from utils.printing import printf, LT


class UIManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ui_config = config.get('processing', {}).get('update_intervals', {})
        
        # UI 컴포넌트들
        self.user_interface: Optional[UserInterface] = None
        self.animation_service: Optional[AnimationService] = None
        
        # UI 업데이트 제어
        self.last_ui_update = 0
        self.last_animation_update = 0
        self.ui_update_interval = self.ui_config.get('ui', 1.0)
        self.animation_update_interval = self.ui_config.get('animation', 0.5)
        
        # UI 통계
        self.ui_stats = {
            'ui_updates': 0,
            'animation_updates': 0,
            'ui_errors': 0,
            'animation_errors': 0
        }
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        try:
            # UserInterface 초기화
            self.user_interface = UserInterface()
            printf("UserInterface initialized", ptype=LT.info)
        except Exception as e:
            printf(f"UserInterface initialization failed: {e}", ptype=LT.warning)
            self.user_interface = None
        
        try:
            # AnimationService 초기화
            self.animation_service = AnimationService(self.config)
            printf("AnimationService initialized", ptype=LT.info)
        except Exception as e:
            printf(f"AnimationService initialization failed: {e}", ptype=LT.warning)
            self.animation_service = None
    
    def update_referee_ui(self, zone_flags: Dict[str, bool]) -> bool:
        current_time = time.perf_counter()
        
        # 업데이트 간격 확인
        if current_time - self.last_ui_update < self.ui_update_interval:
            return False
        
        if not self.user_interface:
            return False
        
        try:
            # 플래그를 UI 형식으로 변환
            flag_list = [
                str(zone_flags.get("On Floor", False)).lower(),
                str(zone_flags.get("Hitted", False)).lower(),
                str(zone_flags.get("Thrower", False)).lower(),
                str(zone_flags.get("OutLined", False)).lower(),
                str(zone_flags.get("L In", False)).lower(),
                str(zone_flags.get("R In", False)).lower(),
                str(zone_flags.get("L Out", False)).lower(),
                str(zone_flags.get("R Out", False)).lower(),
                str(zone_flags.get("Running", False)).lower()
            ]
            
            self.user_interface.update(flag_list)
            self.ui_stats['ui_updates'] += 1
            self.last_ui_update = current_time
            return True
            
        except Exception as e:
            printf(f"UI update error: {e}", ptype=LT.warning)
            self.ui_stats['ui_errors'] += 1
            return False
    
    def update_animation(self, position_data: Dict[float, Any]) -> bool:
        current_time = time.perf_counter()
        
        # 업데이트 간격 확인
        if current_time - self.last_animation_update < self.animation_update_interval:
            return False
        
        if not self.animation_service:
            return False
        
        try:
            success = self.animation_service.update_data(position_data)
            if success:
                self.ui_stats['animation_updates'] += 1
                self.last_animation_update = current_time
            return success
            
        except Exception as e:
            printf(f"Animation update error: {e}", ptype=LT.warning)
            self.ui_stats['animation_errors'] += 1
            return False
    
    def process_animation_updates(self) -> bool:
        if not self.animation_service:
            return False
        
        try:
            self.animation_service.process_updates()
            return True
        except Exception as e:
            printf(f"Animation process error: {e}", ptype=LT.warning)
            return False
    
    def force_animation_update(self, position_data: Dict[float, Any]):
        if self.animation_service:
            try:
                self.animation_service.force_update(position_data)
                printf("Animation force updated", ptype=LT.info)
            except Exception as e:
                printf(f"Force animation update error: {e}", ptype=LT.warning)
    
    def get_ui_statistics(self) -> Dict[str, Any]:
        stats = self.ui_stats.copy()
        
        # 애니메이션 성능 통계 추가
        if self.animation_service:
            animation_stats = self.animation_service.get_performance_stats()
            stats.update({f"animation_{k}": v for k, v in animation_stats.items()})
        
        return stats
    
    def cleanup(self):
        printf("Cleaning up UI components...", ptype=LT.info)
        
        if self.animation_service:
            try:
                self.animation_service.close()
                printf("AnimationService closed", ptype=LT.info)
            except Exception as e:
                printf(f"AnimationService cleanup error: {e}", ptype=LT.warning)
        
        # matplotlib 창들 정리
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            printf(f"Matplotlib cleanup error: {e}", ptype=LT.warning)