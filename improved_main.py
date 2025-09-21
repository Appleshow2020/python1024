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

def main():
    """
    메인 실행 함수:
    1. 설정 로드
    2. 데이터 브로커 및 성능 모니터 초기화
    3. 프로세스 매니저를 통해 각 프로세스 시작
    4. 프로세스 종료 대기
    """
    print("프로그램을 시작합니다...")

    # 1. 설정 관리자 초기화 및 설정 로드
    config_manager = ConfigManager('config.json')
    config = config_manager.get_config()
    if not config:
        print("설정 파일을 로드할 수 없습니다. 프로그램을 종료합니다.")
        return

    # 2. 멀티프로세싱 환경을 위한 공유 데이터 관리자
    with Manager() as manager:
        # 데이터 브로커 및 성능 모니터 초기화
        data_broker = DataBroker(manager)
        performance_monitor = PerformanceMonitor(manager)

        # 3. 프로세스 매니저 초기화
        process_manager = ProcessManager()

        # 각 프로세스를 등록
        # target: 실행할 함수, args: 함수에 전달할 인자
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

        # 4. 모든 등록된 프로세스 시작
        process_manager.start_all()

        try:
            # 메인 프로세스는 모든 자식 프로세스가 끝날 때까지 대기
            while any(p.is_alive() for p in process_manager.get_all_processes()):
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n사용자에 의해 프로그램 종료 요청...")
        finally:
            # 5. 모든 프로세스 종료
            process_manager.stop_all()
            print("모든 프로세스가 안전하게 종료되었습니다.")

if __name__ == '__main__':
    main()