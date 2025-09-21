# detection/detection_process.py
import os
import cv2
import numpy as np
import time
from core.data_broker import DataBroker
from utils.performance_monitor import PerformanceMonitor
from classes.printing import *

def detection_process_main(config: dict, data_broker: DataBroker, perf_monitor: PerformanceMonitor):
    """
    Process that retrieves frames from the data broker, detects the ball, and stores the result back in the broker.
    :param config: Detection settings (hsv_lower, hsv_upper, etc.)
    :param data_broker: Data sharing object
    :param perf_monitor: Performance monitoring object
    """
    printf(f"Detection process started (PID: {os.getpid()})", LT.info)
    
    hsv_lower = np.array(config.get("hsv_lower", [0, 0, 0]))
    hsv_upper = np.array(config.get("hsv_upper", [179, 255, 255]))
    min_radius = config.get("min_radius", 10)
    
    last_frame_timestamp = 0

    try:
        while True:
            current_timestamp = data_broker.get_data("frame_timestamp")

            # Only process when there is a new frame
            if current_timestamp and current_timestamp > last_frame_timestamp:
                frame = data_broker.get_data("latest_frame")
                if frame is None:
                    time.sleep(0.01)
                    continue

                last_frame_timestamp = current_timestamp

                # 1. Preprocessing
                blurred = cv2.GaussianBlur(frame, tuple(config.get("gaussian_blur_kernel", (11, 11))), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                # 2. Create color-based mask
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                # 3. Find contours
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                ball_center = None
                ball_radius = None

                if len(contours) > 0:
                    # Consider the largest contour as the ball
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)

                    if radius > min_radius:
                        ball_center = (int(x), int(y))
                        ball_radius = int(radius)

                # 4. Save detection result
                detection_result = {
                    'center': ball_center,
                    'radius': ball_radius,
                    'timestamp': time.time()
                }
                data_broker.set_data("detection_result", detection_result)
                perf_monitor.update_metric("detection_fps", 1)
            else:
                # If there is no new frame to process, wait briefly
                time.sleep(0.005)

    except KeyboardInterrupt:
        printf("Detection process termination request detected.", LT.info)
    finally:
        printf("Detection process terminated.", LT.info)