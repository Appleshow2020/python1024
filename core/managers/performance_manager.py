# core/managers/performance_manager.py
"""
성능 모니터링 관리자
기존 SystemMonitor, 프로파일링 관련 기능들을 통합 관리
"""

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
    """시스템 성능 모니터링 및 프로파일링 관리자"""
    
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
            printf("psutil available - full system monitoring enabled", LT.info)
        else:
            printf("psutil not available - limited system monitoring", LT.warning)
    
    def _setup_profiling(self) -> bool:
        """프로파일링 설정"""
        # 환경변수 확인
        if os.getenv('PROFILE', 'False').lower() == 'true':
            return True
        
        # 설정 파일 확인
        if self.profiling_config.get('enabled', False):
            return True
        
        return False
    
    def start_profiling(self):
        """메인 프로파일링 시작"""
        if self.enable_profiling:
            self.main_profiler = cProfile.Profile()
            self.main_profiler.enable()
            printf("Main profiling started", LT.info)
    
    def stop_profiling(self):
        """메인 프로파일링 종료"""
        if self.enable_profiling and self.main_profiler:
            self.main_profiler.disable()
            printf("Main profiling stopped", LT.info)
    
    def update_frame_time(self, frame_time: float):
        """프레임 시간 업데이트"""
        self.frame_times.append(frame_time)
    
    def update_system_stats(self):
        """시스템 통계 업데이트"""
        if not self.psutil_available:
            return
            
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_mb)
        except Exception as e:
            printf(f"System stats update failed: {e}", LT.warning)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """종합 성능 리포트 생성"""
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
        """주기적 통계 출력"""
        current_time = time.perf_counter()
        stats_interval = self.config.get('processing', {}).get('update_intervals', {}).get('stats', 5.0)
        
        if current_time - self.last_stats_time < stats_interval:
            return
            
        report = self.get_performance_report()
        
        printf("=== Performance Stats ===", LT.info)
        printf(f"Uptime: {report['uptime']:.1f}s", LT.info)
        printf(f"Total Frames: {report['total_frames']}", LT.info)
        
        if 'avg_fps' in report:
            printf(f"Average FPS: {report['avg_fps']:.1f}", LT.info)
            printf(f"Frame Time: avg={report['avg_frame_time']*1000:.1f}ms, "
                  f"max={report['max_frame_time']*1000:.1f}ms", LT.info)
        
        if 'avg_cpu_usage' in report:
            printf(f"CPU Usage: avg={report['avg_cpu_usage']:.1f}%, "
                  f"max={report['max_cpu_usage']:.1f}%", LT.info)
            printf(f"Memory: avg={report['avg_memory_mb']:.1f}MB, "
                  f"max={report['max_memory_mb']:.1f}MB", LT.info)
        
        self.last_stats_time = current_time
    
    def save_profiling_results(self):
        """프로파일링 결과 저장"""
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
            
            printf(f"Profiling results saved to {profile_file}", LT.info)
            
        except Exception as e:
            printf(f"Failed to save profiling results: {e}", LT.error)
    
    def save_performance_report(self):
        """성능 보고서를 파일로 저장"""
        try:
            report = self.get_performance_report()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
            
            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            printf(f"Performance report saved to {filename}", LT.info)
            
        except Exception as e:
            printf(f"Failed to save performance report: {e}", LT.warning)