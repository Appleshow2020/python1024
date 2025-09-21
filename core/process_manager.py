from multiprocessing import Process
from classes.printing import *

class ProcessManager:
    """
    Class for managing (creating, starting, stopping) multiple processes in the application.
    """
    def __init__(self):
        self.processes = {}

    def add_process(self, name: str, target, args=()):
        """
        Add a process to be managed.
        :param name: Name to identify the process
        :param target: Function to be executed by the process
        :param args: Tuple of arguments to pass to the target function
        """
        if name in self.processes:
            printf(f"Warning: Process with name '{name}' already exists. Overwriting.", LT.warning)

        process = Process(target=target, args=args, name=name)
        process.daemon = True  # Ensure process terminates with the main process
        self.processes[name] = process
        printf(f"Process '{name}' has been registered.", LT.info)

    def start_all(self):
        """Start all registered processes."""
        if not self.processes:
            printf("No processes to start.", LT.warning)
            return

        printf("Starting all processes...", LT.info)
        for name, process in self.processes.items():
            process.start()
            printf(f"'{name}' process started (PID: {process.pid})", LT.info)

    def stop_all(self):
        """Stop all registered processes."""
        printf("Stopping all processes...", LT.info)
        for name, process in self.processes.items():
            if process.is_alive():
                printf(f"Attempting to stop '{name}' process...", LT.info)
                process.terminate() # Send SIGTERM
                process.join(timeout=2) # Wait for termination
                if process.is_alive():
                    printf(f"'{name}' did not terminate cleanly. Forcing kill.", LT.error)
                    process.kill() # Send SIGKILL
                printf(f"'{name}' process has been stopped.", LT.info)

    def get_process(self, name: str) -> Process:
        """Return a specific process object by name."""
        return self.processes.get(name)

    def get_all_processes(self) -> list[Process]:
        """Return a list of all process objects."""
        return list(self.processes.values())