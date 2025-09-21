# core/data_broker.py
from multiprocessing.managers import BaseManager

class DataBroker:
    """
    프로세스 간 데이터 교환을 위한 공유 객체 관리 클래스.
    - 공유 상태(dict)와 메시지 큐(Queue)를 제공합니다.
    """
    def __init__(self, manager: BaseManager):
        """
        DataBroker를 초기화합니다.
        :param manager: multiprocessing.Manager 인스턴스
        """
        # 공유 상태 저장을 위한 딕셔너리
        self._shared_dict = manager.dict()
        # 프로세스 간 메시지 전달을 위한 큐
        self._shared_queue = manager.Queue()

    def set_data(self, key, value):
        """공유 딕셔너리에 데이터를 설정합니다."""
        self._shared_dict[key] = value

    def get_data(self, key, default=None):
        """공유 딕셔너리에서 데이터를 가져옵니다."""
        return self._shared_dict.get(key, default)

    def put_message(self, message):
        """공유 큐에 메시지를 넣습니다."""
        self._shared_queue.put(message)

    def get_message(self, block=True, timeout=None):
        """
        공유 큐에서 메시지를 가져옵니다.
        :param block: 메시지가 있을 때까지 대기할지 여부
        :param timeout: 대기 시간 (초)
        :return: 큐에서 꺼낸 메시지
        """
        if not self._shared_queue.empty():
            return self._shared_queue.get(block, timeout)
        return None