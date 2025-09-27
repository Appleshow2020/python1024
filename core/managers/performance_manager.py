# core/managers/performance_manager.py
import time
import cProfile
import pstats
import os
from collections import deque
from typing import Dict, Any, Optional

from classes.printing import printf, LT

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceManager:
    """
    PerformanceManager is responsible for monitoring and reporting the performance of an application,
    including frame timing, CPU and memory usage, and optional profiling using cProfile.
    Attributes:
        config (Dict[str, Any]): Configuration dictionary for performance and profiling settings.
        profiling_config (Dict[str, Any]): Profiling-specific configuration.
        start_time (float): Timestamp when the manager was initialized.
        frame_times (deque): Recent frame times for FPS and frame time statistics.
        cpu_usage (deque): Recent CPU usage percentages.
        memory_usage (deque): Recent memory usage in MB.
        last_stats_time (float): Last time periodic stats were printed.
        enable_profiling (bool): Whether profiling is enabled.
        main_profiler (Optional[cProfile.Profile]): The main cProfile profiler instance.
        psutil_available (bool): Whether psutil is available for system monitoring.
        psutil (module): Reference to the psutil module (if available).
        process (psutil.Process): Reference to the current process (if psutil is available).
    Methods:
        _setup_profiling() -> bool:
            Determines if profiling should be enabled based on environment variables and config.
        start_profiling():
            Starts the main cProfile profiler if profiling is enabled.
        stop_profiling():
            Stops the main cProfile profiler if profiling is enabled.
        update_frame_time(frame_time: float):
            Records a new frame time for performance statistics.
        update_system_stats():
            Updates CPU and memory usage statistics if psutil is available.
        get_performance_report() -> Dict[str, Any]:
            Returns a dictionary containing current performance statistics.
        print_periodic_stats():
            Prints performance statistics at configured intervals.
        save_profiling_results():
            Saves the cProfile results to file if profiling is enabled.
        save_performance_report():
            Saves the current performance report to a JSON file.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profiling_config = config.get('profiling', {})
        
        # 성능 데이터
        self.start_time = time.perf_counter()
        self.frame_times = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.last_stats_time = 0
        
        # 프로파일링
        self.enable_profiling = self._setup_profiling()
        self.main_profiler: Optional[cProfile.Profile] = None
        
        # psutil 설정
        self.psutil_available = PSUTIL_AVAILABLE
        if self.psutil_available:
            self.psutil = psutil
            self.process = psutil.Process()
            printf("psutil available - full system monitoring enabled", ptype=LT.info)
        else:
            printf("psutil not available - limited system monitoring", ptype=LT.warning)
    
    def _setup_profiling(self) -> bool:
        # 환경변수 확인
        if os.getenv('PROFILE', 'False').lower() == 'true':
            return True
        
        # 설정 파일 확인
        if self.profiling_config.get('enabled', False):
            return True
        
        return False
    
    def start_profiling(self):
        if self.enable_profiling:
            self.main_profiler = cProfile.Profile()
            self.main_profiler.enable()
            printf("Main profiling started", ptype=LT.info)
    
    def stop_profiling(self):
        if self.enable_profiling and self.main_profiler:
            self.main_profiler.disable()
            printf("Main profiling stopped", ptype=LT.info)
    
    def update_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
    
    def update_system_stats(self):
        if not self.psutil_available:
            return
            
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_mb)
        except Exception as e:
            printf(f"System stats update failed: {e}", ptype=LT.warning)
    
    def get_performance_report(self) -> Dict[str, Any]:
        current_time = time.perf_counter()
        uptime = current_time - self.start_time
        
        report = {
            'uptime': uptime,
            'total_frames': len(self.frame_times),
            'profiling_enabled': self.enable_profiling
        }
        
        # 프레임 성능
        if self.frame_times:
            times = list(self.frame_times)
            report.update({
                'avg_fps': len(times) / sum(times) if sum(times) > 0 else 0,
                'avg_frame_time': sum(times) / len(times),
                'max_frame_time': max(times),
                'min_frame_time': min(times)
            })
        
        # 시스템 성능
        if self.psutil_available and self.cpu_usage and self.memory_usage:
            cpu_list = list(self.cpu_usage)
            memory_list = list(self.memory_usage)
            
            report.update({
                'avg_cpu_usage': sum(cpu_list) / len(cpu_list),
                'max_cpu_usage': max(cpu_list),
                'avg_memory_mb': sum(memory_list) / len(memory_list),
                'max_memory_mb': max(memory_list)
            })
        
        return report
    
    def print_periodic_stats(self):
        current_time = time.perf_counter()
        stats_interval = self.config.get('processing', {}).get('update_intervals', {}).get('stats', 5.0)
        
        if current_time - self.last_stats_time < stats_interval:
            return
            
        report = self.get_performance_report()
        
        printf("=== Performance Stats ===", ptype=LT.info)
        printf(f"Uptime: {report['uptime']:.1f}s", ptype=LT.info)
        printf(f"Total Frames: {report['total_frames']}", ptype=LT.info)
        
        if 'avg_fps' in report:
            printf(f"Average FPS: {report['avg_fps']:.1f}", ptype=LT.info)
            printf(f"Frame Time: avg={report['avg_frame_time']*1000:.1f}ms, "
                  f"max={report['max_frame_time']*1000:.1f}ms", ptype=LT.info)
        
        if 'avg_cpu_usage' in report:
            printf(f"CPU Usage: avg={report['avg_cpu_usage']:.1f}%, "
                  f"max={report['max_cpu_usage']:.1f}%", ptype=LT.info)
            printf(f"Memory: avg={report['avg_memory_mb']:.1f}MB, "
                  f"max={report['max_memory_mb']:.1f}MB", ptype=LT.info)
        
        self.last_stats_time = current_time
    
    def save_profiling_results(self):
        if not self.enable_profiling or not self.main_profiler:
            return
        
        try:
            output_dir = self.profiling_config.get('output_dir', 'profiles')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            profile_file = os.path.join(output_dir, f"main_profile_{timestamp}.prof")
            text_file = os.path.join(output_dir, f"main_profile_{timestamp}.txt")
            
            # 프로파일 저장
            self.main_profiler.dump_stats(profile_file)
            
            # 텍스트 리포트 생성
            with open(text_file, 'w') as f:
                stats = pstats.Stats(self.main_profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(file=f)
            
            printf(f"Profiling results saved to {profile_file}", ptype=LT.info)
            
        except Exception as e:
            printf(f"Failed to save profiling results: {e}", ptype=LT.error)
    
    def save_performance_report(self):
        try:
            report = self.get_performance_report()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
            
            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            printf(f"Performance report saved to {filename}", ptype=LT.info)
            
        except Exception as e:
            printf(f"Failed to save performance report: {e}", ptype=LT.warning)