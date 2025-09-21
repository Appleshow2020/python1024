# ui/dashboard_process.py
import os
import time
from core.data_broker import DataBroker
from utils.performance_monitor import PerformanceMonitor

def clear_console():
    """콘솔 화면을 지웁니다."""
    os.system('cls' if os.name == 'nt' else 'clear')

def dashboard_process_main(data_broker: DataBroker, perf_monitor: PerformanceMonitor):
    """
    주기적으로 데이터 브로커와 성능 모니터에서 정보를 가져와 콘솔에 출력하는 프로세스.
    """
    print(f"대시보드 프로세스 시작 (PID: {os.getpid()})")
    
    try:
        while True:
            clear_console()
            print("--- 실시간 모니터링 대시보드 ---")
            
            # 1. 시스템 성능 정보
            cpu_usage = perf_monitor.get_metric('cpu_usage')
            mem_usage = perf_monitor.get_metric('memory_usage_mb')
            print(f"\n[시스템 정보]")
            print(f"  - CPU 사용률: {cpu_usage:.2f}%" if cpu_usage is not None else "  - CPU 사용률: N/A")
            print(f"  - 메모리 사용량: {mem_usage:.2f} MB" if mem_usage is not None else "  - 메모리 사용량: N/A")

            # 2. 프로세스별 FPS
            cam_fps = perf_monitor.get_fps("camera_fps")
            det_fps = perf_monitor.get_fps("detection_fps")
            print(f"\n[프로세스 성능 (FPS)]")
            print(f"  - 카메라: {cam_fps:.2f} FPS")
            print(f"  - 검출: {det_fps:.2f} FPS")

            # 3. 데이터 처리 상태
            camera_status = data_broker.get_data("camera_status", "N/A")
            detection_result = data_broker.get_data("detection_result")
            print(f"\n[데이터 상태]")
            print(f"  - 카메라 상태: {camera_status}")
            
            if detection_result and detection_result['center']:
                center = detection_result['center']
                radius = detection_result['radius']
                print(f"  - 공 검출됨: 좌표=({center[0]}, {center[1]}), 반지름={radius}")
            else:
                print("  - 공 검출 안됨")

            print("\n---------------------------------")
            print("(종료하려면 메인 터미널에서 Ctrl+C를 누르세요)")
            
            time.sleep(0.5) # 2Hz로 업데이트

    except KeyboardInterrupt:
        pass
    finally:
        print("대시보드 프로세스를 종료합니다.")