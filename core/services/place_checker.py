# core/services/place_checker.py
from typing import Dict, Optional, Tuple, List
from core.models.field_zones import FieldZones


class PlaceCheckerService:
    """최적화된 볼 위치 체커 서비스"""
    
    def __init__(self, zones: FieldZones):
        self.zones = zones
        self.flags = {
            "On Floor": False, 
            "Hitted": False, 
            "Thrower": False, 
            "OutLined": False,
            "L In": False, 
            "R In": False, 
            "L Out": False, 
            "R Out": False, 
            "Running": False,
        }
        self.last_position = None
        self.last_result = None
        self.position_cache = {}
        self.cache_threshold = 0.1

    def check(self, bx: float, by: float) -> Optional[str]:
        """볼 위치 체크 - 고급 캐싱"""
        # 캐시 키 생성
        cache_key = (round(bx / self.cache_threshold), round(by / self.cache_threshold))
        
        # 캐시에서 확인
        if cache_key in self.position_cache:
            result = self.position_cache[cache_key]
            self._update_flags(result)
            return result
            
        # 새로운 위치 계산
        result = None
        for key in self.flags:
            self.flags[key] = False

        if self._in_box_fast(bx, by, self.zones.li):
            self.flags["L In"] = True
            result = "li"
        elif self._in_box_fast(bx, by, self.zones.ri):
            self.flags["R In"] = True
            result = "ri"
        elif self._in_box_fast(bx, by, self.zones.lo):
            self.flags["L Out"] = True
            result = "lo"
        elif self._in_box_fast(bx, by, self.zones.ro):
            self.flags["R Out"] = True
            result = "ro"
            
        # 캐시 업데이트 (크기 제한)
        if len(self.position_cache) < 1000:
            self.position_cache[cache_key] = result
            
        return result
    
    def _update_flags(self, result: Optional[str]):
        """플래그 업데이트"""
        for key in self.flags:
            self.flags[key] = False
            
        if result == "li":
            self.flags["L In"] = True
        elif result == "ri":
            self.flags["R In"] = True
        elif result == "lo":
            self.flags["L Out"] = True
        elif result == "ro": 
            self.flags["R Out"] = True
    
    def get_flags(self) -> Dict[str, bool]:
        """현재 플래그 상태 반환"""
        return self.flags.copy()
    
    def get_flags_as_list(self) -> List[str]:
        """UI 업데이트용 플래그 리스트 반환"""
        return [
            str(self.flags["On Floor"]).lower(),
            str(self.flags["Hitted"]).lower(), 
            str(self.flags["Thrower"]).lower(),
            str(self.flags["OutLined"]).lower(),
            str(self.flags["L In"]).lower(),
            str(self.flags["R In"]).lower(),
            str(self.flags["L Out"]).lower(),
            str(self.flags["R Out"]).lower(),
            str(self.flags["Running"]).lower()
        ]
    
    @staticmethod
    def _in_box_fast(x: float, y: float, 
                     box: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """최적화된 박스 내부 확인"""
        (xmin, ymin), (xmax, ymax) = box
        return xmin <= x <= xmax and ymin <= y <= ymax
    
    def clear_cache(self):
        """위치 캐시 초기화"""
        self.position_cache.clear()