# utils/geometry_utils.py
"""
기하학적 계산 유틸리티
기존 build_point_grid_optimized, make_field_zones_optimized 함수들을 분리
"""

from typing import List, Tuple
from core.models.field_zones import FieldZones


def build_point_grid() -> List[Tuple[float, float]]:
    """격자 포인트 생성"""
    pdx = [-11, -4, 4, 11, -8, -4, 4, 8, -8, -4, 4, 8, -11, -4, 4, 11]
    pdy = [7, 7, 7, 7, 4, 4, 4, 4, -4, -4, -4, -4, -7, -7, -7, -7]
    return list(zip(pdx, pdy))


def make_field_zones(point_list: List[Tuple[float, float]]) -> FieldZones:
    """필드 영역 생성"""
    P = point_list
    
    def make_box(p1_idx: int, p2_idx: int):
        """두 포인트로 박스 생성"""
        x1, y1 = P[p1_idx]
        x2, y2 = P[p2_idx] 
        return ((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)))

    return FieldZones(
        li=make_box(4, 0),   # Left In
        ri=make_box(7, 3),   # Right In
        lo=make_box(12, 15), # Left Out
        ro=make_box(8, 11)   # Right Out
    )