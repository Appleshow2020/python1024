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

def main():
    """
    Main execution function:
    1. Load configuration
    2. Initialize data broker and performance monitor
    3. Start each process via process manager
    4. Wait for process termination
    """
    printf("Starting the program...", LT.info)

    # 1. Initialize config manager and load configuration
    config_manager = ConfigManager('config.json')
    config = config_manager.get_config()
    if not config:
        printf("Unable to load the configuration file. Exiting the program.", LT.error)
        return

    # 2. Shared data manager for multiprocessing environment
    with Manager() as manager:
        # Initialize data broker and performance monitor
        data_broker = DataBroker(manager)
        performance_monitor = PerformanceMonitor(manager)

        # 3. Initialize process manager
        process_manager = ProcessManager(use_custom_logging=True)

        # Register each process
        # target: function to execute, args: arguments to pass to the function
        process_manager.add_process(
            name="Camera",
            target=camera_process_main,
            args=(config['camera'], data_broker, performance_monitor)
        )
        process_manager.add_process(
            name="Detection",
            target=detection_process_main,
            args=(config['detection'], data_broker, performance_monitor)
        )
        process_manager.add_process(
            name="Dashboard",
            target=dashboard_process_main,
            args=(data_broker, performance_monitor)
        )

        # 4. Start all registered processes
        process_manager.start_all()

        try:
            # Main process waits until all child processes finish
            while any(p.is_alive() for p in process_manager.get_all_processes()):
                time.sleep(1)
        except KeyboardInterrupt:
            printf("\nProgram termination requested by user...", LT.info)
        finally:
            # 5. Stop all processes
            process_manager.stop_all()
            printf("All processes have been safely terminated.", LT.info)

if __name__ == '__main__':
    main()