from dataclasses import dataclass, asdict

@dataclass
class GyroData:
    """자이로스코프 센서 데이터"""
    camera_id: str  
    timestamp: float
    roll: float
    pitch: float
    yaw: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    motor_pan: float
    motor_tilt: float
    
    def to_dict(self):
        return asdict(self)