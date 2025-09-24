# improved_main.py
import time
from multiprocessing import Manager

from core.config_manager import ConfigManager
from core.data_broker import DataBroker
from core.process_manager import ProcessManager
from camera.camera_process import camera_process_main
from detection.detection_process import detection_process_main
from ui.dashboard_process import dashboard_process_main
from utils.performance_monitor import PerformanceMonitor
from classes.printing import *

def camera_wrapper(camera_config):
    """Camera process wrapper that creates its own DataBroker and PerformanceMonitor"""
    with Manager() as manager:
        data_broker = DataBroker(manager)
        performance_monitor = PerformanceMonitor(manager)
        camera_process_main(camera_config, data_broker, performance_monitor)

def detection_wrapper(detection_config):
    """Detection process wrapper that creates its own DataBroker and PerformanceMonitor"""
    with Manager() as manager:
        data_broker = DataBroker(manager)
        performance_monitor = PerformanceMonitor(manager)
        detection_process_main(detection_config, data_broker, performance_monitor)

def dashboard_wrapper():
    """Dashboard process wrapper that creates its own DataBroker and PerformanceMonitor"""
    with Manager() as manager:
        data_broker = DataBroker(manager)
        performance_monitor = PerformanceMonitor(manager)
        dashboard_process_main(data_broker, performance_monitor)

def main():
    """
    Main execution function:
    1. Load configuration
    2. Initialize process manager
    3. Start each process via process manager with wrapper functions
    4. Wait for process termination
    """
    printf("Starting the program...", LT.info)

    # 1. Initialize config manager and load configuration
    config_manager = ConfigManager('config.json')
    config = config_manager.get_config()
    if not config:
        printf("Unable to load the configuration file. Exiting the program.", LT.error)
        return

    # 2. Initialize process manager
    process_manager = ProcessManager(use_custom_logging=True)

    # Register each process with wrapper functions that handle their own Manager
    process_manager.add_process(
        name="Camera",
        target=camera_wrapper,
        args=(config['camera'],)
    )
    process_manager.add_process(
        name="Detection",
        target=detection_wrapper,
        args=(config['detection'],)
    )
    process_manager.add_process(
        name="Dashboard",
        target=dashboard_wrapper,
        args=()
    )

    # 3. Start all registered processes
    process_manager.start_all()

    try:
        # Main process waits until all child processes finish
        while any(p.is_alive() for p in process_manager.get_all_processes()):
            time.sleep(1)
    except KeyboardInterrupt:
        printf("\nProgram termination requested by user...", LT.info)
    finally:
        # 4. Stop all processes
        process_manager.stop_all()
        printf("All processes have been safely terminated.", LT.info)

if __name__ == '__main__':
    main()