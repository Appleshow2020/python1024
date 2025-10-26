from main.ApplicationController import ApplicationController
from utils.printing import LT, printf
import cv2

class CleanupManager:
    def __init__(self, Self: ApplicationController):
        self.app = Self
    
    def cleanup(self):
        """애플리케이션 정리"""
        printf("=== Starting Application Cleanup ===", ptype=LT.info)

        try:
            self._cleanup_cameras()
            self._cleanup_ui()
            self._cleanup_opencv_windows()
            self._save_profiling_data()
            printf("=== Application cleanup completed successfully ===", ptype=LT.success)

        except Exception as e:
            printf(f"Cleanup error: {e}", ptype=LT.error)
        finally:
            self._print_final_statistics()  # 항상 실행되도록

    def _print_final_statistics(self):
        self._print_runtime_summary()
        self._print_detection_summary()
        self._print_tracking_summary()
        self._print_performance_summary()

    def _print_performance_summary(self):
        if self.app.performance_manager:
            performance_report = self.app.performance_manager.get_performance_report()
            avg_fps = performance_report.get('avg_fps', 0)
            uptime = performance_report.get('uptime', 0)
            printf(f"System Performance: {avg_fps:.1f} FPS average, {uptime:.1f}s uptime", ptype=LT.info)

    def _print_tracking_summary(self):
        if self.app.tracking_manager:
            tracking_stats = self.app.tracking_manager.get_tracking_statistics()
            track_success = tracking_stats.get('success_rate', 0)
            printf(f"Tracking Performance: {track_success:.1f}% triangulation success", ptype=LT.info)

    def _print_detection_summary(self):
        if self.app.detection_manager:
            detection_stats = self.app.detection_manager.get_detection_statistics()
            success_rate = detection_stats.get('success_rate', 0)
            printf(f"Detection Performance: {success_rate:.1f}% success rate", ptype=LT.info)

    def _print_runtime_summary(self):
        if self.app.performance_manager:
            total_uptime = self.app.performance_manager.get_performance_report().get('uptime', 0)
            printf(f"Total Runtime: {self.frame_count} frames processed, {total_uptime:.1f} seconds elapsed", ptype=LT.info)
        else:
            printf(f"Total Runtime: {self.frame_count} frames processed", ptype=LT.info)

    def _save_profiling_data(self):
        if self.app.performance_manager:
            printf("Saving profiling results...", ptype=LT.info)
            self.app.performance_manager.stop_profiling()
            self.app.performance_manager.save_profiling_results()
            self.app.performance_manager.save_performance_report()

        if self.app.detection_manager:
            self.app.detection_manager.save_detection_profile("detection_profile.prof")

    def _cleanup_opencv_windows(self):
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            printf(f"Error while destroying OpenCV windows: {e}", ptype=LT.warning)

    def _cleanup_ui(self):
        if self.app.ui_manager:
            printf("Cleaning up UI components...", ptype=LT.info)
            self.app.ui_manager.cleanup()

    def _cleanup_cameras(self):
        if self.app.camera_manager:
            printf("Stopping camera threads...", ptype=LT.info)
            self.app.camera_manager.stop_cameras()