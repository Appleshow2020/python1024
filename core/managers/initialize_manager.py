from typing import Optional
from core.managers.performance_manager import PerformanceManager
from core.managers.camera_manager import CameraManager
from core.managers.tracking_manager import TrackingManager
from core.managers.detection_manager import DetectionManager
from core.managers.ui_manager import UIManager
from utils.config import ConfigManager
from utils.printing import printf, LT

class InitializeManager:
    def __init__(self):
        self.performance_manager: Optional[PerformanceManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.tracking_manager: Optional[TrackingManager] = None
        self.detection_manager: Optional[DetectionManager] = None
        self.ui_manager: Optional[UIManager] = None
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

    def initialize_performance_manager(self) -> bool:
        printf("Initializing performance manager...", ptype=LT.info)
        self.performance_manager = PerformanceManager(self.config)
        self.performance_manager.start_profiling()
        return True
    
    def initialize_camera_manager(self) -> bool:
        printf("Initializing camera manager...", ptype=LT.info)
        self.camera_manager = CameraManager(self.config)
        # 카메라 검색 및 초기화
        camera_count = self.config['camera'].get('camera_count', None)
        if not self.camera_manager.initialize_cameras(camera_count):
            printf("Failed to initialize cameras", ptype=LT.error)
            return False
        
        # 카메라 스레드 시작
        if not self.camera_manager.start_camera_threads():
            printf("Failed to start camera threads", ptype=LT.error)
            return False

        return True
    
    def initialize_tracking_manager(self) -> bool:
        printf("Initializing tracking manager...", ptype=LT.info)
        self.tracking_manager = TrackingManager(self.config)

        if not self.tracking_manager.initialize_calibration(
            self.camera_manager.selected_cameras
        ):
            printf("Failed to initialize camera calibration", ptype=LT.error)
            return False
        
        return True
    
    def initialize_detection_manager(self) -> bool:
        printf("Initializing detection manager...", ptype=LT.info)
        self.detection_manager = DetectionManager(self.camera_manager, self.config)
        return True
    
    def initialize_ui_manager(self) -> bool:
        printf("Initializing UI manager...", ptype=LT.info)
        self.ui_manager = UIManager(self.config)
        printf("=== All components initialized successfully ===", ptype=LT.success)
        return True