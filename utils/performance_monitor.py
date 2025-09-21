# utils/performance_monitor.py
import psutil
import time
from multiprocessing.managers import BaseManager

class PerformanceMonitor:
    """
    CPU, 메모리 사용량 및 사용자 정의 메트릭(예: FPS)을 모니터링하는 클래스.
    멀티프로세싱 환경에서 안전하게 데이터를 공유합니다.
    """
    def __init__(self, manager: BaseManager):
        """
        PerformanceMonitor를 초기화합니다.
        :param manager: multiprocessing.Manager 인스턴스
        """
        self._metrics = manager.dict()
        self._fps_counters = manager.dict()
        self._fps_start_times = manager.dict()
        self.process = psutil.Process() # 현재 프로세스에 대한 정보를 가져옴

        # 초기 시스템 메트릭 설정
        self.update_system_metrics()

    def update_system_metrics(self):
        """CPU와 메모리 사용량을 업데이트합니다."""
        with self.process.oneshot():
            self._metrics['cpu_usage'] = psutil.cpu_percent(interval=None)
            self._metrics['memory_usage_mb'] = self.process.memory_info().rss / (1024 * 1024)

    def update_metric(self, name: str, value: float):
        """
        사용자 정의 메트릭을 업데이트합니다. FPS 계산에 주로 사용됩니다.
        :param name: 메트릭 이름 (예: "camera_fps")
        :param value: 증가시킬 값 (보통 1)
        """
        if name not in self._fps_counters:
            self._fps_counters[name] = 0
            self._fps_start_times[name] = time.time()
        
        self._fps_counters[name] += value

    def get_metric(self, name: str, default=None):
        """저장된 메트릭 값을 반환합니다."""
        return self._metrics.get(name, default)

    def get_fps(self, name: str) -> float:
        """
        지정된 메트릭의 FPS(초당 프레임 수)를 계산하여 반환합니다.
        :param name: FPS를 계산할 메트릭 이름
        :return: 계산된 FPS 값
        """
        if name not in self._fps_counters:
            return 0.0

        start_time = self._fps_start_times.get(name, 0)
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 1: # 1초 이상 경과했을 때만 업데이트
            count = self._fps_counters.get(name, 0)
            fps = count / elapsed_time
            self._metrics[name + "_val"] = fps # 계산된 fps 값 저장
            # 카운터와 시작 시간 초기화
            self._fps_counters[name] = 0
            self._fps_start_times[name] = time.time()

        return self._metrics.get(name + "_val", 0.0)