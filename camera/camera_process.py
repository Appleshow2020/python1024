# camera/camera_process.py
import os
import cv2
import time
from core.data_broker import DataBroker
from utils.performance_monitor import PerformanceMonitor
from classes.printing import *

def camera_process_main(config: dict, data_broker: DataBroker, perf_monitor: PerformanceMonitor):
    """
    카메라에서 프레임을 지속적으로 캡처하여 데이터 브로커에 저장하는 프로세스.
    :param config: 카메라 설정 (source, width, height, fps)
    :param data_broker: 데이터 공유 객체
    :param perf_monitor: 성능 측정 객체
    """
    printf(f"카메라 프로세스 시작 (PID: {os.getpid()})", LT.info)
    cap = cv2.VideoCapture(config.get("source", 0))
    if not cap.isOpened():
        printf(f"카메라 소스 '{config.get('source')}'를 열 수 없습니다.", LT.error)
        data_broker.set_data("camera_status", "Error")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height", 720))
    cap.set(cv2.CAP_PROP_FPS, config.get("fps", 30))
    printf("카메라 설정 완료.",LT.success)
    data_broker.set_data("camera_status", "Running")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                printf("프레임을 읽을 수 없습니다. 스트림이 끝났을 수 있습니다.", LT.error)
                break

            # 프레임을 데이터 브로커에 저장
            data_broker.set_data("latest_frame", frame)
            data_broker.set_data("frame_timestamp", time.time())
            
            frame_count += 1
            perf_monitor.update_metric("camera_fps", 1) # FPS 계산을 위해 프레임 카운트

            # CPU 부하를 줄이기 위해 약간의 대기 시간 추가 (필요 시)
            # time.sleep(1 / config.get("fps", 30))

    except KeyboardInterrupt:
        printf("카메라 프로세스 종료 요청 감지.", LT.debug)
    finally:
        cap.release()
        data_broker.set_data("camera_status", "Stopped")
        printf("카메라 리소스 해제 완료. 카메라 프로세스를 종료합니다.", LT.info)