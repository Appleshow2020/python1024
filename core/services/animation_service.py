# core/services/animation_service.py
"""
애니메이션 서비스
기존 AdvancedAnimationWrapper 클래스를 서비스로 분리
"""

import time
import queue
import numpy as np
import matplotlib.pyplot as plt
from threading import Event
from collections import deque
from typing import Dict, Any, Optional

from classes.printing import printf, LT


class AnimationService:
    """고급 애니메이션 서비스 - GPU 가속 및 최적화 포함"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.display_config = config.get('display', {})
        self.processing_config = config.get('processing', {})
        
        # 애니메이션 설정
        self.last_update = 0
        self.data_queue = queue.Queue(maxsize=self.processing_config.get('queue_size', 10))
        self.stop_event = Event()
        self.ui_update_event = Event()
        
        # matplotlib 컴포넌트들
        self.fig = None
        self.ax = None
        self.plot_data = {'x': [], 'y': [], 'timestamps': []}
        
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
            
            # 블리팅 활성화로 성능 향상
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)
            self.ax.set_title("Advanced Ball Tracking")
            self.ax.grid(True, alpha=0.3)
            
            printf("Advanced matplotlib initialized", LT.info)
        except Exception as e:
            printf(f"Matplotlib init failed: {e}", LT.error)
            
    def update_data(self, pl: Dict[float, Any]) -> bool:
        """최적화된 데이터 업데이트"""
        current_time = time.perf_counter()
        update_interval = self.processing_config.get('update_intervals', {}).get('animation', 0.5)
        
        if current_time - self.last_update < update_interval:
            return False
            
        try:
            # 큐 관리 - 최신 데이터만 유지
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self.data_queue.put({
                'pl': pl.copy(), 
                'timestamp': current_time,
                'data_points': len(pl)
            })
            self.ui_update_event.set()
            self.last_update = current_time
            return True
        except Exception as e:
            printf(f"Animation update failed: {e}", LT.warning)
        return False
    
    def process_updates(self):
        """고성능 업데이트 처리"""
        if not self.ui_update_event.is_set():
            return
            
        start_time = time.perf_counter()
        
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self._update_plot_advanced(data)
            self.ui_update_event.clear()
            
            # 렌더링 시간 기록
            render_time = time.perf_counter() - start_time
            self.render_times.append(render_time)
            
        except queue.Empty:
            pass
        except Exception as e:
            printf(f"Process updates failed: {e}", LT.warning)
    
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
            
            # 새로운 데이터 그리기
            if len(positions) > 0:
                # 점 그리기
                points = self.ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7, animated=True)
                
                # 궤적 그리기
                if len(positions) > 1:
                    line, = self.ax.plot(x_coords, y_coords, 'r-', alpha=0.5, linewidth=1, animated=True)
                    self.ax.draw_artist(line)
                
                self.ax.draw_artist(points)
            
            # 블리팅으로 업데이트
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
            
        except Exception as e:
            printf(f"Advanced plot update failed: {e}", LT.warning)
    
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
            printf(f"Force update failed: {e}", LT.warning)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """렌더링 성능 통계"""
        if not self.render_times:
            return {}
            
        times = list(self.render_times)
        return {
            'avg_render_time': np.mean(times),
            'max_render_time': np.max(times),
            'render_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def close(self):
        """정리 함수"""
        self.stop_event.set()
        if self.fig is not None:
            plt.close(self.fig)