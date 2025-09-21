# detection/detection_process.py
import os
import cv2
import numpy as np
import time
from core.data_broker import DataBroker
from utils.performance_monitor import PerformanceMonitor

def detection_process_main(config: dict, data_broker: DataBroker, perf_monitor: PerformanceMonitor):
    """
    데이터 브로커에서 프레임을 가져와 공을 검출하고 결과를 다시 브로커에 저장하는 프로세스.
    :param config: 검출 설정 (hsv_lower, hsv_upper 등)
    :param data_broker: 데이터 공유 객체
    :param perf_monitor: 성능 측정 객체
    """
    print(f"검출 프로세스 시작 (PID: {os.getpid()})")
    
    hsv_lower = np.array(config.get("hsv_lower", [0, 0, 0]))
    hsv_upper = np.array(config.get("hsv_upper", [179, 255, 255]))
    min_radius = config.get("min_radius", 10)
    
    last_frame_timestamp = 0

    try:
        while True:
            current_timestamp = data_broker.get_data("frame_timestamp")
            
            # 새로운 프레임이 있을 때만 처리
            if current_timestamp and current_timestamp > last_frame_timestamp:
                frame = data_broker.get_data("latest_frame")
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                last_frame_timestamp = current_timestamp

                # 1. 전처리
                blurred = cv2.GaussianBlur(frame, tuple(config.get("gaussian_blur_kernel", (11, 11))), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                # 2. 색상 기반 마스크 생성
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # 3. 컨투어 검출
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                ball_center = None
                ball_radius = None

                if len(contours) > 0:
                    # 가장 큰 컨투어를 공으로 간주
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    
                    if radius > min_radius:
                        ball_center = (int(x), int(y))
                        ball_radius = int(radius)

                # 4. 검출 결과 저장
                detection_result = {
                    'center': ball_center,
                    'radius': ball_radius,
                    'timestamp': time.time()
                }
                data_broker.set_data("detection_result", detection_result)
                perf_monitor.update_metric("detection_fps", 1)
            else:
                # 처리할 새 프레임이 없으면 잠시 대기
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("검출 프로세스 종료 요청 감지.")
    finally:
        print("검출 프로세스를 종료합니다.")