# core/services/system_monitor.py

import time
import numpy as np
from collections import deque
from typing import Dict, Any, Optional
from utils.printing import printf, LT

# psutil 선택적 import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemMonitorService:
    """시스템 성능 모니터링 서비스"""
    
    def __init__(self):
        self.start_time = time.perf_counter()
        self.frame_times = deque(maxlen=1000)
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.last_stats_time = 0
        
        # psutil 설정
        self.psutil_available = PSUTIL_AVAILABLE
        if self.psutil_available:
            try:
                self.psutil = psutil
                self.process = psutil.Process()
                printf("System monitoring with psutil enabled", LT.info)
            except Exception as e:
                self.psutil_available = False
                printf(f"psutil initialization failed: {e}", LT.warning)
        else:
            printf("psutil not available - limited system monitoring", LT.warning)
    
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
        """성능 리포트 생성"""
        current_time = time.perf_counter()
        uptime = current_time - self.start_time
        
        report = {
            'uptime': uptime,
            'total_frames': len(self.frame_times),
            'psutil_available': self.psutil_available
        }
        
        # 프레임 성능 통계
        if self.frame_times:
            times = list(self.frame_times)
            total_time = sum(times)
            if total_time > 0:
                report.update({
                    'avg_fps': len(times) / total_time,
                    'avg_frame_time': np.mean(times),
                    'max_frame_time': np.max(times),
                    'min_frame_time': np.min(times),
                    'frame_time_std': np.std(times)
                })
        
        # 시스템 리소스 통계
        if self.psutil_available and self.cpu_usage and self.memory_usage:
            cpu_list = list(self.cpu_usage)
            memory_list = list(self.memory_usage)
            
            report.update({
                'avg_cpu_usage': np.mean(cpu_list),
                'max_cpu_usage': np.max(cpu_list),
                'min_cpu_usage': np.min(cpu_list),
                'avg_memory_mb': np.mean(memory_list),
                'max_memory_mb': np.max(memory_list),
                'min_memory_mb': np.min(memory_list)
            })
        
        return report
    
    def get_current_system_info(self) -> Dict[str, Any]:
        """현재 시스템 정보"""
        info = {
            'timestamp': time.perf_counter(),
            'uptime': time.perf_counter() - self.start_time
        }
        
        if self.psutil_available:
            try:
                info.update({
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                    'threads': self.process.num_threads(),
                    'open_files': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
                })
            except Exception as e:
                printf(f"Failed to get current system info: {e}", LT.warning)
        
        return info
    
    def print_stats(self, stats_interval: float = 5.0):
        """주기적 통계 출력"""
        current_time = time.perf_counter()
        
        if current_time - self.last_stats_time < stats_interval:
            return
            
        report = self.get_performance_report()
        
        printf("=== System Performance Stats ===", LT.info)
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
    
    def reset_stats(self):
        """통계 초기화"""
        self.frame_times.clear()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.start_time = time.perf_counter()
        printf("System monitor stats reset", LT.info)
    
    def is_performance_degraded(self, fps_threshold: float = 15.0, 
                              cpu_threshold: float = 80.0, 
                              memory_threshold_mb: float = 1000.0) -> Dict[str, bool]:
        """성능 저하 감지"""
        issues = {
            'low_fps': False,
            'high_cpu': False,
            'high_memory': False
        }
        
        # FPS 확인
        if self.frame_times and len(self.frame_times) > 10:
            recent_times = list(self.frame_times)[-10:]
            current_fps = 10 / sum(recent_times) if sum(recent_times) > 0 else 0
            if current_fps < fps_threshold:
                issues['low_fps'] = True
        
        # CPU 사용률 확인
        if self.cpu_usage and len(self.cpu_usage) > 5:
            recent_cpu = list(self.cpu_usage)[-5:]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            if avg_cpu > cpu_threshold:
                issues['high_cpu'] = True
        
        # 메모리 사용량 확인
        if self.memory_usage and len(self.memory_usage) > 5:
            recent_memory = list(self.memory_usage)[-5:]
            avg_memory = sum(recent_memory) / len(recent_memory)
            if avg_memory > memory_threshold_mb:
                issues['high_memory'] = True
        
        return issues