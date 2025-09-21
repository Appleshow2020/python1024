# camera/camera_process.py
import os
import cv2
import time
from core.data_broker import DataBroker
from utils.performance_monitor import PerformanceMonitor
from classes.printing import *

def camera_process_main(config: dict, data_broker: DataBroker, perf_monitor: PerformanceMonitor):
    """
    Process that continuously captures frames from the camera and stores them in the data broker.
    :param config: Camera settings (source, width, height, fps)
    :param data_broker: Data sharing object
    :param perf_monitor: Performance monitoring object
    """
    printf(f"Camera process started (PID: {os.getpid()})", LT.info)
    cap = cv2.VideoCapture(config.get("source", 0))
    if not cap.isOpened():
        printf(f"Cannot open camera source '{config.get('source')}'", LT.error)
        data_broker.set_data("camera_status", "Error")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height", 720))
    cap.set(cv2.CAP_PROP_FPS, config.get("fps", 30))
    printf("Camera settings applied.", LT.success)
    data_broker.set_data("camera_status", "Running")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                printf("Cannot read frame. The stream may have ended.", LT.error)
                break

            # Store the frame in the data broker
            data_broker.set_data("latest_frame", frame)
            data_broker.set_data("frame_timestamp", time.time())
            
            frame_count += 1
            perf_monitor.update_metric("camera_fps", 1) # Count for FPS calculation

            # Add a short delay to reduce CPU load (if needed)
            # time.sleep(1 / config.get("fps", 30))

    except KeyboardInterrupt:
        printf("Camera process termination request detected.", LT.debug)
    finally:
        cap.release()
        data_broker.set_data("camera_status", "Stopped")
        printf("Camera resources released. Camera process terminated.", LT.info)