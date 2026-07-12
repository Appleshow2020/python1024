from core.services.sensor_controller import SensorController

def main(cameras_config):
    # 컨트롤러 생성
    # controller = CentralGyroController()
    
    # # 카메라 추가 (예제 - 실제 IP로 변경 필요)
    # cameras_config = [
    #     {'id': 'cam_bottom_1', 'ip': '192.168.1.101', 'port': 5001},
    #     {'id': 'cam_bottom_2', 'ip': '192.168.1.102', 'port': 5001},
    #     {'id': 'cam_bottom_3', 'ip': '192.168.1.103', 'port': 5001},
    #     {'id': 'cam_bottom_4', 'ip': '192.168.1.104', 'port': 5001},
    #     {'id': 'cam_bottom_5', 'ip': '192.168.1.105', 'port': 5001},
    #     {'id': 'cam_bottom_6', 'ip': '192.168.1.106', 'port': 5001},
    #     {'id': 'cam_ceiling', 'ip': '192.168.1.107', 'port': 5001},
    # ]
    
    print("\n카메라 연결 중...")
    for cam_config in cameras_config:
        controller.add_camera(
            cam_config['id'],
            cam_config['ip'],
            cam_config['port']
        )
    
    try:
        # while True:
        #     print("\n" + "="*80)
        #     print("메뉴")
        #     print("="*80)
        #     print("1. 상태 확인")
        #     print("2. 특정 카메라 확인")
        #     print("3. 특정 카메라 조정")
        #     print("4. 모든 카메라 조정")
        #     print("5. 자동 모니터링 시작")
        #     print("6. 자동 모니터링 중지")
        #     print("7. 기준 자세 재설정")
        #     print("8. 설정 저장")
        #     print("9. 종료")
        #
        # choice = input("\n선택: ").strip()
        
        if choice == '1':
            controller.print_status()
        
        elif choice == '2':
            camera_id = input("카메라 ID: ").strip()
            result = controller.check_camera_orientation(camera_id)
            if result:
                print(f"\n조정 필요: {result['needs_adjustment']}")
                print(f"오차: {result['errors']}")
        
        elif choice == '3':
            camera_id = input("카메라 ID: ").strip()
            controller.adjust_camera(camera_id)
        
        elif choice == '4':
            for camera_id in controller.receivers.keys():
                controller.adjust_camera(camera_id)
        
        elif choice == '5':
            controller.start_montoring()
        
        elif choice == '6':
            controller.stop_monitoring()
        
        elif choice == '7':
            camera_id = input("카메라 ID: ").strip()
            controller.set_reference_orientation(camera_id, use_current=True)
        
    
    except KeyboardInterrupt:
        print("\n\n프로그램 종료 중...")
    
    finally:
        controller.cleanup()
