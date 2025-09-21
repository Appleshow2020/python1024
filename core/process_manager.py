# core/process_manager.py
from multiprocessing import Process

class ProcessManager:
    """
    어플리케이션의 여러 프로세스를 관리(생성, 시작, 종료)하는 클래스.
    """
    def __init__(self):
        self.processes = {}

    def add_process(self, name: str, target, args=()):
        """
        관리할 프로세스를 추가합니다.
        :param name: 프로세스를 식별할 이름
        :param target: 프로세스가 실행할 함수
        :param args: target 함수에 전달될 인자 튜플
        """
        if name in self.processes:
            print(f"경고: '{name}' 이름의 프로세스가 이미 존재합니다. 덮어씁니다.")

        process = Process(target=target, args=args, name=name)
        process.daemon = True  # 메인 프로세스 종료 시 함께 종료되도록 설정
        self.processes[name] = process
        print(f"프로세스 '{name}'이(가) 등록되었습니다.")

    def start_all(self):
        """등록된 모든 프로세스를 시작합니다."""
        if not self.processes:
            print("시작할 프로세스가 없습니다.")
            return

        print("모든 프로세스를 시작합니다...")
        for name, process in self.processes.items():
            process.start()
            print(f" - '{name}' 프로세스 시작 (PID: {process.pid})")

    def stop_all(self):
        """등록된 모든 프로세스를 종료합니다."""
        print("모든 프로세스를 종료합니다...")
        for name, process in self.processes.items():
            if process.is_alive():
                print(f" - '{name}' 프로세스 종료 시도...")
                process.terminate() # SIGTERM 전송
                process.join(timeout=2) # 종료 대기
                if process.is_alive():
                    print(f"   ! '{name}'이(가) 정상적으로 종료되지 않아 강제 종료합니다.")
                    process.kill() # SIGKILL 전송
                print(f" - '{name}' 프로세스가 종료되었습니다.")

    def get_process(self, name: str) -> Process:
        """이름으로 특정 프로세스 객체를 반환합니다."""
        return self.processes.get(name)

    def get_all_processes(self) -> list[Process]:
        """모든 프로세스 객체 리스트를 반환합니다."""
        return list(self.processes.values())