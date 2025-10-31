# core/services/animation_service.py
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Event
from collections import deque
from typing import Dict, Any

from utils.printing import printf, LT


class AnimationService:
    """고급 애니메이션 서비스 - GPU 가속 및 최적화 포함"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.display_config = config.get('display', {})
        self.processing_config = config.get('processing', {})
        
        # 애니메이션 설정
        self.last_update = 0
        self.data_queue = deque(maxlen=1)   # 최신 데이터만 유지
        self.stop_event = Event()
        self.ui_update_event = Event()
        
        # matplotlib 컴포넌트들
        self.fig = None
        self.ax = None
        self.line = None
        self.points = None
        
        # 성능 모니터링
        self.render_times = deque(maxlen=100)
        
        # 초기화
        self._init_matplotlib()
        
    def _init_matplotlib(self):
        """matplotlib 초기화 및 최적화"""
        try:
            plt.ion()
            figsize = self.display_config.get('plot_size', [8, 6])
            self.fig, self.ax = plt.subplots(figsize=figsize)

            # 초기 plot 객체 생성 (재사용 가능)
            (self.line,) = self.ax.plot([], [], 'r-', alpha=0.5, linewidth=1, animated=True)
            self.points = self.ax.scatter([], [], c='red', s=30, alpha=0.7, animated=True)

            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)
            self.ax.set_title("Advanced Ball Tracking")
            self.ax.grid(True, alpha=0.3)

            printf("Advanced matplotlib initialized", ptype=LT.info)
        except Exception as e:
            printf(f"Matplotlib init failed: {e}", ptype=LT.error)
            
    def update_data(self, pl: Dict[float, Any]) -> bool:
        """최적화된 데이터 업데이트"""
        current_time = time.perf_counter()
        update_interval = self.processing_config.get('update_intervals', {}).get('animation', 0.5)
        
        if current_time - self.last_update < update_interval:
            return False
            
        try:
            # 최신 데이터만 유지
            self.data_queue.clear()
            self.data_queue.append({
                'pl': pl.copy(), 
                'timestamp': current_time,
                'data_points': len(pl)
            })
            self.ui_update_event.set()
            self.last_update = current_time
            return True
        except Exception as e:
            printf(f"Animation update failed: {e}", ptype=LT.warning)
        return False
    
    def process_updates(self):
        """고성능 업데이트 처리"""
        if not self.ui_update_event.is_set():
            return
            
        start_time = time.perf_counter()
        
        try:
            # 창 크기 변경 등에 대비하여 배경 최신화
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            
            while self.data_queue:
                data = self.data_queue.popleft()
                self._update_plot_advanced(data)
            self.ui_update_event.clear()
            
            # 렌더링 시간 기록
            render_time = time.perf_counter() - start_time
            self.render_times.append(render_time)
            
        except Exception as e:
            printf(f"Process updates failed: {e}", ptype=LT.warning)
    
    def _update_plot_advanced(self, data: Dict[str, Any]):
        """고급 플롯 업데이트 - 블리팅 사용"""
        if self.fig is None or self.ax is None:
            return
            
        try:
            pl = data.get('pl', {})
            if not pl:
                return
                
            max_points = self.display_config.get('max_plot_points', 10)
            positions = list(pl.values())[-max_points:]
            timestamps = list(pl.keys())[-max_points:]
            
            if not positions:
                return
                
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # 배경 복원 (블리팅)
            self.fig.canvas.restore_region(self.background)
            
            # 기존 객체 업데이트
            self.line.set_data(x_coords, y_coords)
            self.points.set_offsets(np.c_[x_coords, y_coords])
            
            self.ax.draw_artist(self.line)
            self.ax.draw_artist(self.points)
            
            # 블리팅으로 업데이트
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
            
        except Exception as e:
            printf(f"Advanced plot update failed: {e}", ptype=LT.warning)
    
    def force_update(self, pl: Dict[float, Any]):
        """강제 업데이트 (즉시 실행)"""
        try:
            data = {
                'pl': pl.copy(),
                'timestamp': time.perf_counter(),
                'data_points': len(pl)
            }
            self._update_plot_advanced(data)
        except Exception as e:
            printf(f"Force update failed: {e}", ptype=LT.warning)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """렌더링 성능 통계"""
        if not self.render_times:
            return {}
            
        times = list(self.render_times)
        return {
            'avg_render_time': float(np.mean(times)),
            'max_render_time': float(np.max(times)),
            'render_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def close(self):
        """정리 함수"""
        self.stop_event.set()
        if self.fig is not None:
            plt.close(self.fig)
