from multiprocessing import Process
import logging
import time
import signal
import atexit
from typing import Callable, Tuple, Dict, List, Optional

# 표준 로깅 설정 (pickle 안전)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessManager:
    """
    Class for managing (creating, starting, stopping) multiple processes in the application.
    Provides safe process lifecycle management with proper cleanup.
    """
    def __init__(self, use_custom_logging: bool = False):
        self.processes: Dict[str, Process] = {}
        self.use_custom_logging = use_custom_logging
        
        # 메인 프로세스 종료 시 자동 정리
        atexit.register(self._cleanup_on_exit)
        
        # SIGINT, SIGTERM 핸들러 설정 (Unix/Linux)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (AttributeError, ValueError):
            # Windows에서는 SIGTERM이 없을 수 있음
            pass

    def _signal_handler(self, signum, frame):
        """시그널 핸들러 - 프로세스 정리"""
        self._log(f"Received signal {signum}. Cleaning up processes...", "warning")
        self.stop_all()

    def _cleanup_on_exit(self):
        """종료 시 자동 정리"""
        if self.processes:
            self._log("Application exiting. Cleaning up processes...", "info")
            self.stop_all()

    def _log(self, message: str, level: str = "info"):
        """안전한 로깅 함수"""
        if self.use_custom_logging:
            try:
                # 런타임에 임포트하여 pickle 문제 방지
                from classes.printing import printf, LT
                level_map = {
                    "info": LT.info,
                    "warning": LT.warning,
                    "error": LT.error,
                    "debug": getattr(LT, 'debug', LT.info)
                }
                printf(message, level_map.get(level, LT.info))
            except (ImportError, AttributeError) as e:
                # fallback to standard logging
                getattr(logger, level, logger.info)(f"{message} (Custom logging failed: {e})")
        else:
            getattr(logger, level, logger.info)(message)

    def add_process(self, name: str, target: Callable, args: Tuple = (), kwargs: Dict = None, daemon: bool = True):
        """
        Add a process to be managed.
        
        Args:
            name: Name to identify the process
            target: Function to be executed by the process
            args: Tuple of arguments to pass to the target function
            kwargs: Dictionary of keyword arguments to pass to the target function
        """
        if kwargs is None:
            kwargs = {}
            
        if name in self.processes:
            if self.processes[name].is_alive():
                self._log(f"Warning: Active process with name '{name}' already exists. Stopping it first.", "warning")
                self.stop_process(name)
            else:
                self._log(f"Warning: Dead process with name '{name}' found. Replacing it.", "warning")

        try:
            process = Process(target=target, args=args, kwargs=kwargs, name=name,daemon=daemon)
            process.daemon = True  # Ensure process terminates with the main process
            self.processes[name] = process
            self._log(f"Process '{name}' has been registered.", "info")
        except Exception as e:
            self._log(f"Failed to create process '{name}': {e}", "error")
            raise

    def start_process(self, name: str) -> bool:
        """
        Start a specific process by name.
        
        Args:
            name: Name of the process to start
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if name not in self.processes:
            self._log(f"Process '{name}' not found.", "error")
            return False
            
        process = self.processes[name]
        
        if process.is_alive():
            self._log(f"Process '{name}' is already running.", "warning")
            return True
            
        try:
            process.start()
            self._log(f"'{name}' process started (PID: {process.pid})", "info")
            return True
        except Exception as e:
            self._log(f"Failed to start process '{name}': {e}", "error")
            return False

    def start_all(self) -> int:
        """
        Start all registered processes.
        
        Returns:
            int: Number of processes successfully started
        """
        if not self.processes:
            self._log("No processes to start.", "warning")
            return 0

        self._log("Starting all processes...", "info")
        started_count = 0
        
        for name in self.processes.keys():
            if self.start_process(name):
                started_count += 1
                
        self._log(f"Started {started_count}/{len(self.processes)} processes.", "info")
        return started_count

    def stop_process(self, name: str, timeout: float = 5.0) -> bool:
        """
        Stop a specific process by name.
        
        Args:
            name: Name of the process to stop
            timeout: Time to wait for graceful termination
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if name not in self.processes:
            self._log(f"Process '{name}' not found.", "error")
            return False
            
        process = self.processes[name]
        
        if not process.is_alive():
            self._log(f"Process '{name}' is not running.", "info")
            return True
            
        try:
            self._log(f"Attempting to stop '{name}' process (PID: {process.pid})...", "info")
            
            # 1단계: 정상 종료 시도
            process.terminate()
            process.join(timeout=timeout)
            
            # 2단계: 강제 종료가 필요한 경우
            if process.is_alive():
                self._log(f"'{name}' did not terminate gracefully. Forcing kill.", "warning")
                process.kill()
                process.join(timeout=2.0)
                
            if process.is_alive():
                self._log(f"Failed to kill process '{name}' (PID: {process.pid})", "error")
                return False
            else:
                self._log(f"'{name}' process has been stopped.", "info")
                return True
                
        except Exception as e:
            self._log(f"Error stopping process '{name}': {e}", "error")
            return False

    def stop_all(self, timeout: float = 5.0) -> int:
        """
        Stop all registered processes.
        
        Args:
            timeout: Time to wait for each process to terminate gracefully
            
        Returns:
            int: Number of processes successfully stopped
        """
        if not self.processes:
            self._log("No processes to stop.", "info")
            return 0
            
        self._log("Stopping all processes...", "info")
        stopped_count = 0
        
        for name in list(self.processes.keys()):
            if self.stop_process(name, timeout):
                stopped_count += 1
                
        self._log(f"Stopped {stopped_count}/{len(self.processes)} processes.", "info")
        return stopped_count

    def restart_process(self, name: str, timeout: float = 5.0) -> bool:
        """
        Restart a specific process.
        
        Args:
            name: Name of the process to restart
            timeout: Time to wait for termination
            
        Returns:
            bool: True if restarted successfully, False otherwise
        """
        if name not in self.processes:
            self._log(f"Process '{name}' not found.", "error")
            return False
            
        # 기존 프로세스 정보 저장
        old_process = self.processes[name]
        target = old_process._target
        args = old_process._args
        kwargs = old_process._kwargs or {}
        
        # 프로세스 중지
        if not self.stop_process(name, timeout):
            return False
            
        # 새 프로세스 생성 및 시작
        self.add_process(name, target, args, kwargs)
        return self.start_process(name)

    def get_process(self, name: str) -> Optional[Process]:
        """Return a specific process object by name."""
        return self.processes.get(name)

    def get_all_processes(self) -> List[Process]:
        """Return a list of all process objects."""
        return list(self.processes.values())

    def get_process_status(self, name: str) -> Optional[str]:
        """
        Get the status of a specific process.
        
        Returns:
            str: 'alive', 'dead', or None if process not found
        """
        if name not in self.processes:
            return None
        return 'alive' if self.processes[name].is_alive() else 'dead'

    def get_all_statuses(self) -> Dict[str, str]:
        """Get status of all processes."""
        return {name: self.get_process_status(name) for name in self.processes}

    def remove_process(self, name: str, stop_if_running: bool = True) -> bool:
        """
        Remove a process from management.
        
        Args:
            name: Name of the process to remove
            stop_if_running: Whether to stop the process if it's running
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        if name not in self.processes:
            self._log(f"Process '{name}' not found.", "error")
            return False
            
        process = self.processes[name]
        
        if process.is_alive():
            if stop_if_running:
                if not self.stop_process(name):
                    self._log(f"Failed to stop process '{name}' before removal.", "error")
                    return False
            else:
                self._log(f"Process '{name}' is still running. Cannot remove.", "error")
                return False
                
        del self.processes[name]
        self._log(f"Process '{name}' has been removed from management.", "info")
        return True

    def wait_for_all(self, timeout: Optional[float] = None):
        """
        Wait for all processes to complete.
        
        Args:
            timeout: Maximum time to wait (None for no timeout)
        """
        start_time = time.time()
        
        for name, process in self.processes.items():
            if process.is_alive():
                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                    
                self._log(f"Waiting for process '{name}' to complete...", "info")
                process.join(timeout=remaining_timeout)
                
                if process.is_alive():
                    self._log(f"Process '{name}' did not complete within timeout.", "warning")
