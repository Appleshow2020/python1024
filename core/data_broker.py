# core/data_broker.py
from multiprocessing.managers import BaseManager

class DataBroker:
    """
    Shared object management class for inter-process data exchange.
    - Provides shared state (dict) and message queue (Queue).
    """
    def __init__(self, manager: BaseManager):
        """
        Initialize DataBroker.
        :param manager: multiprocessing.Manager instance
        """
        # Dictionary for storing shared state
        self._shared_dict = manager.dict()
        # Queue for inter-process message passing
        self._shared_queue = manager.Queue()

    def set_data(self, key, value):
        """Set data in the shared dictionary."""
        self._shared_dict[key] = value

    def get_data(self, key, default=None):
        """Get data from the shared dictionary."""
        return self._shared_dict.get(key, default)

    def put_message(self, message):
        """Put a message into the shared queue."""
        self._shared_queue.put(message)

    def get_message(self, block=True, timeout=None):
        """
        Get a message from the shared queue.
        :param block: Whether to wait until a message is available
        :param timeout: Wait time (seconds)
        :return: Message taken from the queue
        """
        if not self._shared_queue.empty():
            return self._shared_queue.get(block, timeout)
        return None