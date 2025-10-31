from core.managers.cleanup_manager import CleanupManager
from core.managers.statistics_manager import StatisticsManager
from core.managers.initialize_manager import InitializeManager
from utils.config import ConfigManager
from utils.printing import LT, printf
import cv2
import signal
import time
import traceback
import datetime

class ApplicationController:
    def __init__(self):
        # 설정 관리
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.initialize_manager = InitializeManager(self)
        self.cleanup_manager = CleanupManager(self)
        self.statistics_manager = StatisticsManager(self)

        self.KEY_QUIT        = ord('q')
        self.KEY_STATS       = ord('s')
        self.KEY_RESET       = ord('r')
        self.KEY_DEBUG       = ord('d')
        self.KEY_PLOT_UPDATE = ord('p')
        
        self.DEFAULT_FPS                     = self.config.get('camera', {}).get('fps', 30)
        self.LOG_INTERVAL_SECONDS            = self.config.get('performance', {}).get('stats_interval', 3.0)
        self.UI_UPDATE_FRAME_INTERVAL        = self.config.get('ui', {}).get('update_interval', 15)
        self.FRAME_DROP_THRESHOLD_MULTIPLIER = self.config.get('performance', {}).get('frame_drop_threshold_multiplier', 2.0)
        self.MIN_CAMERAS_FOR_TRACKING        = self.config.get('tracking', {}).get('min_cameras_for_tracking', 2)

        self.camera_manager = self.initialize_manager.camera_manager
        self.detection_manager = self.initialize_manager.detection_manager  
        self.tracking_manager = self.initialize_manager.tracking_manager
        self.ui_manager = self.initialize_manager.ui_manager
        self.performance_manager = self.initialize_manager.performance_manager
        self.image_manager = self.initialize_manager.image_manager
        self.data_manager = self.initialize_manager.data_manager
        
        self.sensor_controller = self.initialize_manager.sensor_controller

        # 애플리케이션 상태
        self.is_running = False
        self.frame_count = 0

        self._setup_signal_handlers()
        
        printf("Application Controller initialized", ptype=LT.info)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """시스템 신호 처리"""
        printf(f"Received signal {signum}, shutting down gracefully...", ptype=LT.warning)
        self.is_running = False

    def initialize(self) -> bool:
        """애플리케이션 초기화"""
        printf("=== Initializing Advanced Ball Tracking System ===", ptype=LT.info)

        try:
            return (
                self.initialize_manager.initialize_performance_manager()
            and self.initialize_manager.initialize_camera_manager()
            and self.initialize_manager.initialize_tracking_manager()
            and self.initialize_manager.initialize_detection_manager()
            and self.initialize_manager.initialize_ui_manager()
            and self.initialize_manager.initialize_image_manager()
            and self.initialize_manager.initialize_data_manager()
            and self.initialize_manager.initialize_sensor_controller()
            )
        
        except Exception as e:
            printf(f"Initialization failed: {e}", ptype=LT.error)
            traceback.print_exc()
            return False
    
    def run(self) -> bool:
        """메인 애플리케이션 실행"""
        if not self.initialize():
            printf("Application initialization failed", ptype=LT.error)
            return False

        return self._main_loop()

    def _main_loop(self) -> bool:
        """메인 프로세싱 루프"""
        printf("=== Starting Main Processing Loop ===", ptype=LT.info)
        self._print_controls()

        self.is_running = True
        frame_interval = self._get_frame_interval()
        last_log_time = time.perf_counter()

        try:
            while self.is_running:
                loop_start = time.perf_counter()
                self.frame_count += 1

                tracking_result = self._process_frame_pipeline(loop_start)

                if self._should_log(last_log_time):
                    self._log_periodic_status(tracking_result)
                    last_log_time = time.perf_counter()

                self._control_frame_rate(loop_start, frame_interval)

                if not self._handle_user_input():
                    break

            printf("Main processing loop completed", ptype=LT.info)
            return True

        except KeyboardInterrupt:
            printf("Application terminated by user", ptype=LT.info)
            return True
        
        except Exception as e:
            printf(f"Main loop error: {e}", ptype=LT.error)
            traceback.print_exc()
            return False
        
    def _get_frame_interval(self) -> float:
        """프레임 간격 계산"""
        target_fps = self.config.get('camera', {}).get('fps', None)
        if not isinstance(target_fps, (int, float)) or target_fps is None or target_fps <= 0:
            printf("Invalid or missing target_fps value in configuration. Using default value 30.", ptype=LT.warning)
            target_fps = self.DEFAULT_FPS

        return 1.0 / target_fps
    
    def _process_frame_pipeline(self, loop_start: float):
        self._update_performance_monitoring(loop_start)
        pts_2d, cam_ids, frame = self._perform_ball_detection()
        tracking_result = self._perform_3d_tracking(pts_2d, cam_ids)
        self._save_tracking_data(tracking_result)
        self._update_user_interfaces(tracking_result)
        self._save_frame_and_record(cam_ids[0], frame) if cam_ids else None
        self._delete_old_images()

        return tracking_result

    def _initialize_sensor_controller(self):
        self.sensor_controller = self.sensor_controller
        for cam_config in self.sensors_config:
            self.sensor_controller.add_sensor(
                cam_config['id'],
                cam_config['ip'],
                cam_config['port']
            )   

    def _save_tracking_data(self, tracking_result):
        if not (self.data_manager and tracking_result):
            return
        tracking_result_return = {
            'timestamp': tracking_result.get('position_entry', {}).get('timestamp', datetime.datetime.now()),
            'position': tracking_result.get('position_3d', (0,0,0)),
            'velocity': tracking_result.get('position_entry', {}).get('velocity', (0,0,0)),
            'direction': tracking_result.get('position_entry', {}).get('direction', (0,0,0)),
            'zone_info': tracking_result.get('zone_info', ''),
            'detection_count': tracking_result.get('detection_count', 0)
        }
        
        self.data_manager.save_tracking_data(tracking_result_return)

    def _save_frame_and_record(self, camera_id, frame):
        if not self.image_manager:
            return
        self.image_manager.save_frame_and_record(camera_id, frame)

    def _delete_old_images(self):
        if not self.image_manager:
            return
        self.image_manager.delete_old_images()

    def _save_tracking_data(self, tracking_result):
        if not (self.data_manager and tracking_result):
            return
        self.data_manager.save_tracking_data(tracking_result)
    
    def _should_log(self, last_log_time: float) -> bool:
        """주기적 상태 로깅 여부 결정"""
        return (time.perf_counter() - last_log_time) >= self.LOG_INTERVAL_SECONDS
    
    def _print_controls(self):
        """컨트롤 안내 출력"""
        printf("=== Controls ===", ptype=LT.info)
        printf("'q' = Quit application", ptype=LT.info)
        printf("'s' = Show detailed statistics", ptype=LT.info)
        printf("'r' = Reset all statistics and data", ptype=LT.info)
        printf("'d' = Show debug information", ptype=LT.info)
        printf("'p' = Force plot update", ptype=LT.info)
        printf("================", ptype=LT.info)

    def _update_performance_monitoring(self, loop_start: float):
        """성능 모니터링 업데이트"""
        if not self.performance_manager:
            return
        
        frame_time = time.perf_counter() - loop_start
        self.performance_manager.update_frame_time(frame_time)
        self.performance_manager.update_system_stats()
        self.performance_manager.print_periodic_stats()

    def _perform_ball_detection(self) -> tuple:
        """볼 검출 수행"""
        if not self.detection_manager:
            return [], [], []

        try:
            return self.detection_manager.process_frame_detections()
        except Exception as e:
            printf(f"Ball detection error: {e}", ptype=LT.error)
            return [], [], []

    def _perform_3d_tracking(self, pts_2d: list, cam_ids: list) -> dict | None:
        """3D 트래킹 수행"""
        if not self.tracking_manager or len(pts_2d) < self.MIN_CAMERAS_FOR_TRACKING:
            return None

        try:
            return self.tracking_manager.process_detections(pts_2d, cam_ids)
        
        except Exception as e:
            printf(f"3D tracking error: {e}", ptype=LT.error)
            return None
        
    def _update_user_interfaces(self, tracking_result):
        """사용자 인터페이스 업데이트"""
        if not self.ui_manager or tracking_result is None:
            return

        try:
            # 레퍼리 UI 업데이트
            self._update_referee_ui(tracking_result)

            # 애니메이션 프로세싱 (메인 스레드에서)
            self._update_animation_if_needed()

        except Exception as e:
            printf(f"UI update error: {e}", ptype=LT.warning)

    def _update_referee_ui(self, tracking_result):
        zone_info = tracking_result.get('zone_info', None)
        if not zone_info:
            return

        zone_flags = zone_info['flags']
        self.ui_manager.update_referee_ui(zone_flags)
        position_data = self.tracking_manager.get_position_data_for_animation()
        self.ui_manager.update_animation(position_data)

    def _update_animation_if_needed(self):
        if self.frame_count % self.UI_UPDATE_FRAME_INTERVAL == 0:
            self.ui_manager.process_animation_updates()

    def _handle_user_input(self) -> bool:
        """사용자 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:
            return True
        
        key_handlers = {
            self.KEY_QUIT: self._handle_quit,
            self.KEY_STATS: self._show_detailed_statistics,
            self.KEY_RESET: self._reset_all_data,   
            self.KEY_DEBUG: self._show_debug_information,
            self.KEY_PLOT_UPDATE: self._force_plot_update
        }
        handler = key_handlers.get(key, None)
        if handler:
            return handler() if handler == self._handle_quit else (handler() or True)

        return True
    
    def _handle_quit(self) -> bool:
        printf("Quit requested by user", ptype=LT.info)
        self.is_running = False
        return False

    def _log_periodic_status(self, tracking_result):
        """주기적 상태 로깅"""
        if not tracking_result:
            return
        
        position_3d = tracking_result.get('position_3d', [0.0, 0.0, 0.0])
        confidence = tracking_result.get('position_entry', {}).get('confidence', 0.0)
        
        printf(f"Ball Position: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}), "
              f"Confidence: {confidence:.2f}", 
              ptype=LT.success)

    def _control_frame_rate(self, loop_start: float, frame_interval: float):
        """프레임 속도 제한"""
        elapsed = time.perf_counter() - loop_start

        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        elif elapsed > frame_interval * 2:
            printf(f"Frame drop detected: processing took {elapsed*1000:.1f}ms", ptype=LT.warning)

    def _show_detailed_statistics(self):
        """상세 통계 표시"""
        self.statistics_manager.show_detailed_statistics()
            
    def _reset_all_data(self):
        printf("Resetting all statistics and data...", ptype=LT.info)

        if self.detection_manager:
            self.detection_manager.reset_statistics()

        if self.tracking_manager:
            self.tracking_manager.reset_tracking_data()

        self.frame_count = 0
        printf("All data reset completed", ptype=LT.success)

    def _show_debug_information(self):
        """디버그 정보 표시"""
        printf("=== Debug Information ===", ptype=LT.debug)
        printf(f"Application Status: Running={self.is_running}, Frames={self.frame_count}", ptype=LT.debug)

        if not self.camera_manager:
            return
        
        camera_stats = self.camera_manager.get_camera_stats()
        for cam_id, stats in camera_stats.items():
            printf(f"Camera {cam_id}: FPS={stats.get('avg_fps', 0):.1f}, "
                  f"Captured={stats.get('frames_captured', 0)}", ptype=LT.debug)

    def _force_plot_update(self):
        """플롯 강제 업데이트"""
        if not (self.ui_manager and self.tracking_manager):
            return
        try:
            position_data = self.tracking_manager.get_position_data_for_animation()
            self.ui_manager.force_animation_update(position_data)
            printf("Plot force updated", ptype=LT.info)
        except Exception as e:
            printf(f"Force plot update failed: {e}", ptype=LT.warning)

    def cleanup(self):
        self.cleanup_manager.cleanup(self)