# core/models/field_zones.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FieldZones:
    """필드 영역 정의 데이터 모델"""
    li: Tuple[Tuple[float, float], Tuple[float, float]]  # Left In
    ri: Tuple[Tuple[float, float], Tuple[float, float]]  # Right In
    lo: Tuple[Tuple[float, float], Tuple[float, float]]  # Left Out
    ro: Tuple[Tuple[float, float], Tuple[float, float]]  # Right Out
    
    def get_zone_by_name(self, zone_name: str):
        """이름으로 존 반환"""
        zone_map = {
            'li': self.li,
            'ri': self.ri,
            'lo': self.lo,
            'ro': self.ro
        }
        return zone_map.get(zone_name.lower())
    
    def get_all_zones(self) -> dict:
        """모든 존 반환"""
        return {
            'li': self.li,
            'ri': self.ri,
            'lo': self.lo,
            'ro': self.ro
        }