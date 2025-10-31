from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CameraStatus:
    """카메라 상태 정보"""
    camera_id: str
    ip_address: str
    port: int
    connected: bool
    last_update: float
    reference_orientation: Optional[Dict[str, float]]
    current_orientation: Optional[Dict[str, float]]
    needs_adjustment: bool
    error: Optional[str]