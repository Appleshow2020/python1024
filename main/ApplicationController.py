from classes.printing import LT, printf
from core.managers.camera_manager import CameraManager
from core.managers.detection_manager import DetectionManager
from core.managers.performance_manager import PerformanceManager
from core.managers.tracking_manager import TrackingManager
from core.managers.ui_manager import UIManager
from utils.config import ConfigManager
import cv2
import signal
import time
import traceback
from typing import Optional, Dict, Any

class ApplicationController:
    """볼 트래킹 시스템의 메인 애플리케이션 컨트롤러"""

    def __init__(self):
        # 설정 관리
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

        # 매니저들 초기화
        self.camera_manager: Optional[CameraManager] = None
        self.detection_manager: Optional[DetectionManager] = None
        self.tracking_manager: Optional[TrackingManager] = None
        self.ui_manager: Optional[UIManager] = None
        self.performance_manager: Optional[PerformanceManager] = None

        # 애플리케이션 상태
        self.is_running = False
        self.frame_count = 0

        # 신호 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        printf("Application Controller initialized", LT.info)

    def _signal_handler(self, signum, frame):
        """시스템 신호 처리"""
        printf(f"Received signal {signum}, shutting down gracefully...", LT.warning)
        self.is_running = False

    def initialize(self) -> bool:
        """애플리케이션 초기화"""
        printf("=== Initializing Advanced Ball Tracking System ===", LT.info)

        try:
            # 1. 성능 매니저 초기화 및 프로파일링 시작
            printf("Initializing performance manager...", LT.info)
            self.performance_manager = PerformanceManager(self.config)
            self.performance_manager.start_profiling()

            # 2. 카메라 매니저 초기화
            printf("Initializing camera manager...", LT.info)
            self.camera_manager = CameraManager(self.config)

            # 카메라 검색 및 초기화
            if not self.camera_manager.initialize_cameras():
                printf("Failed to initialize cameras", LT.error)
                return False

            # 카메라 스레드 시작
            if not self.camera_manager.start_camera_threads():
                printf("Failed to start camera threads", LT.error)
                return False

            # 3. 트래킹 매니저 초기화
            printf("Initializing tracking manager...", LT.info)
            self.tracking_manager = TrackingManager(self.config)

            # 카메라 캘리브레이션
            if not self.tracking_manager.initialize_calibration(
                self.camera_manager.selected_cameras
            ):
                printf("Failed to initialize camera calibration", LT.error)
                return False

            # 4. 검출 매니저 초기화
            printf("Initializing detection manager...", LT.info)
            self.detection_manager = DetectionManager(self.camera_manager, self.config)

            # 5. UI 매니저 초기화
            printf("Initializing UI manager...", LT.info)
            self.ui_manager = UIManager(self.config)

            printf("=== All components initialized successfully ===", LT.success)
            return True

        except Exception as e:
            printf(f"Initialization failed: {e}", LT.error)
            import traceback
            traceback.print_exc()
            return False

    def run(self) -> bool:
        """메인 애플리케이션 실행"""
        if not self.initialize():
            printf("Application initialization failed", LT.error)
            return False

        return self._main_loop()

    def _main_loop(self) -> bool:
        """메인 프로세싱 루프"""
        printf("=== Starting Main Processing Loop ===", LT.info)
        self._print_controls()

        self.is_running = True

        # 타이밍 설정
        target_fps = self.config['camera']['fps']
        frame_interval = 1.0 / target_fps

        # 통계 변수
        last_log_time = time.perf_counter()
        log_interval = 3.0

        try:
            while self.is_running:
                loop_start = time.perf_counter()
                self.frame_count += 1

                # 1. 성능 모니터링
                self._update_performance_monitoring(loop_start)

                # 2. 볼 검출 수행
                pts_2d, cam_ids = self._perform_ball_detection()

                # 3. 3D 트래킹 수행
                tracking_result = self._perform_3d_tracking(pts_2d, cam_ids)

                # 4. UI 업데이트
                self._update_user_interfaces(tracking_result)

                # 5. 키 입력 처리
                if not self._handle_user_input():
                    break

                # 6. 주기적 상태 로깅
                current_time = time.perf_counter()
                if current_time - last_log_time > log_interval:
                    self._log_periodic_status(tracking_result)
                    last_log_time = current_time

                # 7. 프레임 속도 제한
                self._control_frame_rate(loop_start, frame_interval)

            printf("Main processing loop completed", LT.info)
            return True

        except KeyboardInterrupt:
            printf("Application terminated by user", LT.info)
            return True
        except Exception as e:
            printf(f"Main loop error: {e}", LT.error)
            import traceback
            traceback.print_exc()
            return False

    def _print_controls(self):
        """컨트롤 안내 출력"""
        printf("=== Controls ===", LT.info)
        printf("'q' = Quit application", LT.info)
        printf("'s' = Show detailed statistics", LT.info)
        printf("'r' = Reset all statistics and data", LT.info)
        printf("'d' = Show debug information", LT.info)
        printf("'p' = Force plot update", LT.info)
        printf("================", LT.info)

    def _update_performance_monitoring(self, loop_start: float):
        """성능 모니터링 업데이트"""
        if self.performance_manager:
            frame_time = time.perf_counter() - loop_start
            self.performance_manager.update_frame_time(frame_time)
            self.performance_manager.update_system_stats()
            self.performance_manager.print_periodic_stats()

    def _perform_ball_detection(self) -> tuple:
        """볼 검출 수행"""
        if not self.detection_manager:
            return [], []

        try:
            return self.detection_manager.process_frame_detections()
        except Exception as e:
            printf(f"Ball detection error: {e}", LT.error)
            return [], []

    def _perform_3d_tracking(self, pts_2d: list, cam_ids: list):
        """3D 트래킹 수행"""
        if not self.tracking_manager or len(pts_2d) < 2:
            return None

        try:
            return self.tracking_manager.process_detections(pts_2d, cam_ids)
        except Exception as e:
            printf(f"3D tracking error: {e}", LT.error)
            return None

    def _update_user_interfaces(self, tracking_result):
        """사용자 인터페이스 업데이트"""
        if not self.ui_manager or not tracking_result:
            return

        try:
            # 레퍼리 UI 업데이트
            if tracking_result.get('zone_info'):
                zone_flags = tracking_result['zone_info']['flags']
                self.ui_manager.update_referee_ui(zone_flags)

            # 애니메이션 업데이트 (주기적으로만)
            if self.frame_count % 20 == 0:
                position_data = self.tracking_manager.get_position_data_for_animation()
                self.ui_manager.update_animation(position_data)

            # 애니메이션 프로세싱 (메인 스레드에서)
            if self.frame_count % 15 == 0:
                self.ui_manager.process_animation_updates()

        except Exception as e:
            printf(f"UI update error: {e}", LT.warning)

    def _handle_user_input(self) -> bool:
        """사용자 입력 처리"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            printf("Quit requested by user", LT.info)
            self.is_running = False
            return False

        elif key == ord('s'):
            self._show_detailed_statistics()

        elif key == ord('r'):
            self._reset_all_data()

        elif key == ord('d'):
            self._show_debug_information()

        elif key == ord('p'):
            self._force_plot_update()

        return True

    def _log_periodic_status(self, tracking_result):
        """주기적 상태 로깅"""
        if tracking_result:
            position_3d = tracking_result['position_3d']
            confidence = tracking_result['position_entry']['confidence']
            printf(f"Ball Position: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f}), "
                  f"Confidence: {confidence:.2f}", LT.success)

    def _control_frame_rate(self, loop_start: float, frame_interval: float):
        """프레임 속도 제한"""
        elapsed = time.perf_counter() - loop_start

        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        elif elapsed > frame_interval * 2:
            printf(f"Frame drop detected: processing took {elapsed*1000:.1f}ms", LT.warning)

    def _show_detailed_statistics(self):
        """상세 통계 표시"""
        printf("=== Detailed System Statistics ===", LT.info)

        # 검출 통계
        if self.detection_manager:
            detection_stats = self.detection_manager.get_detection_statistics()
            printf(f"Detection Stats - Success Rate: {detection_stats.get('success_rate', 0):.1f}%, "
                  f"Total: {detection_stats.get('total_detections', 0)}", LT.info)

        # 트래킹 통계
        if self.tracking_manager:
            tracking_stats = self.tracking_manager.get_tracking_statistics()
            printf(f"Tracking Stats - Success Rate: {tracking_stats.get('success_rate', 0):.1f}%, "
                  f"Positions: {tracking_stats.get('position_history_size', 0)}", LT.info)

        # UI 통계
        if self.ui_manager:
            ui_stats = self.ui_manager.get_ui_statistics()
            printf(f"UI Stats - Updates: {ui_stats.get('ui_updates', 0)}, "
                  f"Animation: {ui_stats.get('animation_updates', 0)}", LT.info)

        # 성능 통계
        if self.performance_manager:
            perf_stats = self.performance_manager.get_performance_report()
            printf(f"Performance - FPS: {perf_stats.get('avg_fps', 0):.1f}, "
                  f"CPU: {perf_stats.get('avg_cpu_usage', 0):.1f}%, "
                  f"Memory: {perf_stats.get('avg_memory_mb', 0):.1f}MB", LT.info)

    def _reset_all_data(self):
        """모든 데이터 및 통계 리셋"""
        printf("Resetting all statistics and data...", LT.info)

        if self.detection_manager:
            self.detection_manager.reset_statistics()

        if self.tracking_manager:
            self.tracking_manager.reset_tracking_data()

        self.frame_count = 0
        printf("All data reset completed", LT.success)

    def _show_debug_information(self):
        """디버그 정보 표시"""
        printf("=== Debug Information ===", LT.debug)
        printf(f"Application Status: Running={self.is_running}, Frames={self.frame_count}", LT.debug)

        if self.camera_manager:
            camera_stats = self.camera_manager.get_camera_stats()
            for cam_id, stats in camera_stats.items():
                printf(f"Camera {cam_id}: FPS={stats.get('avg_fps', 0):.1f}, "
                      f"Captured={stats.get('frames_captured', 0)}", LT.debug)

    def _force_plot_update(self):
        """플롯 강제 업데이트"""
        if self.ui_manager and self.tracking_manager:
            try:
                position_data = self.tracking_manager.get_position_data_for_animation()
                self.ui_manager.force_animation_update(position_data)
                printf("Plot force updated", LT.info)
            except Exception as e:
                printf(f"Force plot update failed: {e}", LT.warning)

    def cleanup(self):
        """애플리케이션 정리"""
        printf("=== Starting Application Cleanup ===", LT.info)

        try:
            # 1. 카메라 스레드 정지
            if self.camera_manager:
                printf("Stopping camera threads...", LT.info)
                self.camera_manager.stop_cameras()

            # 2. UI 정리
            if self.ui_manager:
                printf("Cleaning up UI components...", LT.info)
                self.ui_manager.cleanup()

            # 3. OpenCV 창들 정리
            cv2.destroyAllWindows()

            # 4. 프로파일링 결과 저장
            if self.performance_manager:
                printf("Saving profiling results...", LT.info)
                self.performance_manager.stop_profiling()
                self.performance_manager.save_profiling_results()
                self.performance_manager.save_performance_report()

            # 5. 검출기 프로파일 저장
            if self.detection_manager:
                self.detection_manager.save_detection_profile("detection_profile.prof")

            # 6. 최종 통계 출력
            self._print_final_statistics()

            printf("=== Application cleanup completed successfully ===", LT.success)

        except Exception as e:
            printf(f"Cleanup error: {e}", LT.error)

    def _print_final_statistics(self):
        """최종 통계 출력"""
        printf("=== Final System Statistics ===", LT.info)
        printf(f"Total Runtime: {self.frame_count} frames processed", LT.info)

        if self.detection_manager:
            detection_stats = self.detection_manager.get_detection_statistics()
            success_rate = detection_stats.get('success_rate', 0)
            printf(f"Detection Performance: {success_rate:.1f}% success rate", LT.info)

        if self.tracking_manager:
            tracking_stats = self.tracking_manager.get_tracking_statistics()
            track_success = tracking_stats.get('success_rate', 0)
            printf(f"Tracking Performance: {track_success:.1f}% triangulation success", LT.info)

        if self.performance_manager:
            performance_report = self.performance_manager.get_performance_report()
            avg_fps = performance_report.get('avg_fps', 0)
            uptime = performance_report.get('uptime', 0)
            printf(f"System Performance: {avg_fps:.1f} FPS average, {uptime:.1f}s uptime", LT.info)