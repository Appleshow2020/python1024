
import cv2
import signal
import time
import traceback

from core.managers.cleanup_manager import CleanupManager
from core.managers.statistics_manager import StatisticsManager
from core.managers.initialize_manager import InitializeManager
from utils.config import ConfigManager
from utils.printing import LT, printf


class ApplicationController:

    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()

        self.initialize_manager = InitializeManager()
        self.cleanup_manager = CleanupManager(self)
        self.statistics_manager = StatisticsManager(self)

        self.KEY_QUIT = ord('q')
        self.KEY_STATS = ord('s')
        self.KEY_RESET = ord('r')
        self.KEY_DEBUG = ord('d')
        self.KEY_PLOT_UPDATE = ord('p')

        self.DEFAULT_FPS = self.config.get(
            'camera', {}
        ).get('fps', 30)

        self.LOG_INTERVAL_SECONDS = self.config.get(
            'performance', {}
        ).get('stats_interval', 3.0)

        self.UI_UPDATE_FRAME_INTERVAL = self.config.get(
            'ui', {}
        ).get('update_interval', 15)

        self.FRAME_DROP_THRESHOLD_MULTIPLIER = self.config.get(
            'performance', {}
        ).get('frame_drop_threshold_multiplier', 2.0)

        self.MIN_CAMERAS_FOR_TRACKING = self.config.get(
            'tracking', {}
        ).get('min_cameras_for_tracking', 2)

        self.is_running = False
        self.frame_count = 0

        self._key_handlers = {
            self.KEY_STATS: self._show_detailed_statistics,
            self.KEY_RESET: self._reset_all_data,
            self.KEY_DEBUG: self._show_debug_information,
            self.KEY_PLOT_UPDATE: self._force_plot_update
        }

        self._setup_signal_handlers()

        printf("Application Controller initialized", ptype=LT.info)

    # --------------------------------------------------
    # Manager accessors
    # --------------------------------------------------

    @property
    def camera_manager(self):
        return self.initialize_manager.camera_manager

    @property
    def detection_manager(self):
        return self.initialize_manager.detection_manager

    @property
    def tracking_manager(self):
        return self.initialize_manager.tracking_manager

    @property
    def ui_manager(self):
        return self.initialize_manager.ui_manager

    @property
    def performance_manager(self):
        return self.initialize_manager.performance_manager

    @property
    def image_manager(self):
        return self.initialize_manager.image_manager

    @property
    def data_manager(self):
        return self.initialize_manager.data_manager

    @property
    def sensor_controller(self):
        return self.initialize_manager.sensor_controller

    # --------------------------------------------------
    # Signal handling
    # --------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        printf(
            f"Received signal {signum}, shutting down gracefully...",
            ptype=LT.warning
        )
        self.is_running = False

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def initialize(self) -> bool:
        printf(
            "=== Initializing Advanced Ball Tracking System ===",
            ptype=LT.info
        )

        try:
            return self._initialize_components()

        except Exception as exc:
            printf(f"Initialization failed: {exc}", ptype=LT.error)
            traceback.print_exc()
            return False

    def _initialize_components(self) -> bool:

        initializers = (
            self.initialize_manager.initialize_performance_manager,
            self.initialize_manager.initialize_camera_manager,
            self.initialize_manager.initialize_tracking_manager,
            self.initialize_manager.initialize_detection_manager,
            self.initialize_manager.initialize_ui_manager,
            self.initialize_manager.initialize_image_manager,
            self.initialize_manager.initialize_data_manager,
            self.initialize_manager.initialize_sensor_controller
        )

        for initialize in initializers:
            if not initialize():
                return False

        return True

    # --------------------------------------------------
    # Application execution
    # --------------------------------------------------

    def run(self) -> bool:
        if not self.initialize():
            printf(
                "Application initialization failed",
                ptype=LT.error
            )
            return False

        return self._main_loop()

    def _main_loop(self) -> bool:
        printf(
            "=== Starting Main Processing Loop ===",
            ptype=LT.info
        )

        self._print_controls()

        self.is_running = True

        frame_interval = self._get_frame_interval()
        last_log_time = time.perf_counter()

        try:
            while self.is_running:
                loop_start = time.perf_counter()
                self.frame_count += 1

                tracking_result = self._process_frame_pipeline(
                    loop_start
                )

                if self._should_log(last_log_time):
                    self._log_periodic_status(tracking_result)
                    last_log_time = time.perf_counter()

                self._control_frame_rate(
                    loop_start,
                    frame_interval
                )

                if not self._handle_user_input():
                    break

            printf(
                "Main processing loop completed",
                ptype=LT.info
            )
            return True

        except KeyboardInterrupt:
            printf(
                "Application terminated by user",
                ptype=LT.info
            )
            return True

        except Exception as exc:
            printf(f"Main loop error: {exc}", ptype=LT.error)
            traceback.print_exc()
            return False

        finally:
            self.is_running = False

    # --------------------------------------------------
    # Frame processing
    # --------------------------------------------------

    def _process_frame_pipeline(self, loop_start: float):
        self._update_performance_monitoring(loop_start)

        pts_2d, cam_ids, frame = self._perform_ball_detection()

        tracking_result = self._perform_3d_tracking(
            pts_2d,
            cam_ids
        )

        self._save_tracking_data(tracking_result)
        self._update_user_interfaces(tracking_result)

        if cam_ids:
            self._save_frame_and_record(cam_ids[0], frame)

        self._delete_old_images()

        return tracking_result

    # --------------------------------------------------
    # Data management
    # --------------------------------------------------

    def _save_tracking_data(self, tracking_result) -> None:
        if self.data_manager is None or tracking_result is None:
            return

        self.data_manager.save_tracking_data(tracking_result)

    def _save_frame_and_record(self, camera_id, frame) -> None:
        if self.image_manager is None:
            return

        self.image_manager.save_frame_and_record(
            camera_id,
            frame
        )

    def _delete_old_images(self) -> None:
        if self.image_manager is None:
            return

        self.image_manager.delete_old_images()

    # --------------------------------------------------
    # Performance management
    # --------------------------------------------------

    def _get_frame_interval(self) -> float:
        target_fps = self.config.get(
            'camera', {}
        ).get('fps', None)

        if (
            not isinstance(target_fps, (int, float))
            or isinstance(target_fps, bool)
            or target_fps <= 0
        ):
            printf(
                "Invalid target_fps. Using default value 30.",
                ptype=LT.warning
            )
            target_fps = 30

        return 1.0 / target_fps

    def _should_log(self, last_log_time: float) -> bool:
        return (
            time.perf_counter() - last_log_time
            >= self.LOG_INTERVAL_SECONDS
        )

    def _update_performance_monitoring(
        self,
        loop_start: float
    ) -> None:

        manager = self.performance_manager

        if manager is None:
            return

        frame_time = time.perf_counter() - loop_start

        manager.update_frame_time(frame_time)
        manager.update_system_stats()
        manager.print_periodic_stats()

    def _control_frame_rate(
        self,
        loop_start: float,
        frame_interval: float
    ) -> None:

        elapsed = time.perf_counter() - loop_start

        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        elif elapsed > (
            frame_interval * self.FRAME_DROP_THRESHOLD_MULTIPLIER
        ):
            printf(
                f"Frame drop detected: processing took "
                f"{elapsed * 1000:.1f}ms",
                ptype=LT.warning
            )

    # --------------------------------------------------
    # Detection and tracking
    # --------------------------------------------------

    def _perform_ball_detection(self) -> tuple:
        manager = self.detection_manager

        if manager is None:
            return [], [], None

        try:
            return manager.process_frame_detections()

        except Exception as exc:
            printf(
                f"Ball detection error: {exc}",
                ptype=LT.error
            )
            return [], [], None

    def _perform_3d_tracking(
        self,
        pts_2d: list,
        cam_ids: list
    ) -> dict | None:

        manager = self.tracking_manager

        if (
            manager is None
            or len(pts_2d) < self.MIN_CAMERAS_FOR_TRACKING
        ):
            return None

        try:
            return manager.process_detections(
                pts_2d,
                cam_ids
            )

        except Exception as exc:
            printf(
                f"3D tracking error: {exc}",
                ptype=LT.error
            )
            return None

    # --------------------------------------------------
    # User interface
    # --------------------------------------------------

    def _print_controls(self) -> None:
        controls = (
            "=== Controls ===",
            "'q' = Quit application",
            "'s' = Show detailed statistics",
            "'r' = Reset all statistics and data",
            "'d' = Show debug information",
            "'p' = Force plot update",
            "================"
        )

        for control in controls:
            printf(control, ptype=LT.info)

    def _update_user_interfaces(self, tracking_result) -> None:
        manager = self.ui_manager

        if manager is None or tracking_result is None:
            return

        try:
            self._update_referee_ui(tracking_result)
            self._update_animation_if_needed()

        except Exception as exc:
            printf(
                f"UI update error: {exc}",
                ptype=LT.warning
            )

    def _update_referee_ui(self, tracking_result) -> None:
        zone_info = tracking_result.get('zone_info')

        if not zone_info:
            return

        zone_flags = zone_info['flags']

        self.ui_manager.update_referee_ui(zone_flags)

        position_data = (
            self.tracking_manager.get_position_data_for_animation()
        )

        self.ui_manager.update_animation(position_data)

    def _update_animation_if_needed(self) -> None:
        interval = self.UI_UPDATE_FRAME_INTERVAL

        if interval > 0 and self.frame_count % interval == 0:
            self.ui_manager.process_animation_updates()

    # --------------------------------------------------
    # Input handling
    # --------------------------------------------------

    def _handle_user_input(self) -> bool:
        key = cv2.waitKey(1) & 0xFF

        if key == 0xFF:
            return True

        if key == self.KEY_QUIT:
            return self._handle_quit()

        handler = self._key_handlers.get(key)

        if handler is not None:
            handler()

        return True

    def _handle_quit(self) -> bool:
        printf(
            "Quit requested by user",
            ptype=LT.info
        )

        self.is_running = False
        return False

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def _log_periodic_status(self, tracking_result) -> None:
        if not tracking_result:
            return

        position = tracking_result.get(
            'position_3d',
            [0.0, 0.0, 0.0]
        )

        confidence = tracking_result.get(
            'position_entry', {}
        ).get('confidence', 0.0)

        printf(
            f"Ball Position: "
            f"({position[0]:.2f}, "
            f"{position[1]:.2f}, "
            f"{position[2]:.2f}), "
            f"Confidence: {confidence:.2f}",
            ptype=LT.success
        )

    def _show_detailed_statistics(self) -> None:
        self.statistics_manager.show_detailed_statistics()

    def _reset_all_data(self) -> None:
        printf(
            "Resetting all statistics and data...",
            ptype=LT.info
        )

        if self.detection_manager is not None:
            self.detection_manager.reset_statistics()

        if self.tracking_manager is not None:
            self.tracking_manager.reset_tracking_data()

        self.frame_count = 0

        printf(
            "All data reset completed",
            ptype=LT.success
        )

    # --------------------------------------------------
    # Debugging
    # --------------------------------------------------

    def _show_debug_information(self) -> None:
        printf(
            "=== Debug Information ===",
            ptype=LT.debug
        )

        printf(
            f"Application Status: "
            f"Running={self.is_running}, "
            f"Frames={self.frame_count}",
            ptype=LT.debug
        )

        manager = self.camera_manager

        if manager is None:
            return

        camera_stats = manager.get_camera_stats()

        for cam_id, stats in camera_stats.items():
            printf(
                f"Camera {cam_id}: "
                f"FPS={stats.get('avg_fps', 0):.1f}, "
                f"Captured={stats.get('frames_captured', 0)}",
                ptype=LT.debug
            )

    def _force_plot_update(self) -> None:
        if not (self.ui_manager and self.tracking_manager):
            return

        try:
            position_data = (
                self.tracking_manager.get_position_data_for_animation()
            )

            self.ui_manager.force_animation_update(position_data)

            printf(
                "Plot force updated",
                ptype=LT.info
            )

        except Exception as exc:
            printf(
                f"Force plot update failed: {exc}",
                ptype=LT.warning
            )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def cleanup(self) -> None:
        self.is_running = False
        self.cleanup_manager.cleanup()
