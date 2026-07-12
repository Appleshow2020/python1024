from utils.printing import LT, printf

class StatisticsManager():
    def __init__(self, Self):
        self.app = Self

    def show_detailed_statistics(self):
        """상세 통계 표시"""
        printf("=== Detailed System Statistics ===", ptype=LT.info)
        self.print_detection_stats()
        self.print_tracking_stats()
        self.print_ui_stats()
        self.print_performance_stats()

    def print_detection_stats(self):
        if not self.app.detection_manager:
            return
        detection_stats = self.app.detection_manager.get_detection_statistics()
        printf(f"Detection Stats - Success Rate: {detection_stats.get('success_rate', 0):.1f}%, "
              f"Total: {detection_stats.get('total_detections', 0)}", ptype=LT.info)
        
    def print_tracking_stats(self):
        if not self.app.tracking_manager:
            return
        tracking_stats = self.app.tracking_manager.get_tracking_statistics()
        printf(f"Tracking Stats - Success Rate: {tracking_stats.get('success_rate', 0):.1f}%, "
              f"Positions: {tracking_stats.get('position_history_size', 0)}", ptype=LT.info)
        
    def print_ui_stats(self):
        if not self.app.ui_manager:
            return
        ui_stats = self.app.ui_manager.get_ui_statistics()
        printf(f"UI Stats - Updates: {ui_stats.get('ui_updates', 0)}, "
              f"Animation: {ui_stats.get('animation_updates', 0)}", ptype=LT.info)
        
    def print_performance_stats(self):
        if not self.app.performance_manager:
            return
        perf_stats = self.app.performance_manager.get_performance_report()
        printf(f"Performance - FPS: {perf_stats.get('avg_fps', 0):.1f}, "
              f"CPU: {perf_stats.get('avg_cpu_usage', 0):.1f}%, "
              f"Memory: {perf_stats.get('avg_memory_mb', 0):.1f}MB", ptype=LT.info)