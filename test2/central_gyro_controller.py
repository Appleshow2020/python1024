import threading
import time
import yaml
from typing import Dict, Optional
from utils.printing import printf, LT
from core.models.camera_status import CameraStatus
from core.models.gyro_data import GyroData
from test2.camera_gyro_receiver import CameraGyroReceiver

class CentralGyroController:
    """
    중앙 처리 장치 - 모든 카메라의 자이로 데이터를 관리하고 자세를 제어
    """
    
    def __init__(self, config_file='camera_gyro_config.yaml'):
        """
        Args:
            config_file: 카메라 설정 파일
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.receivers: Dict[str, CameraGyroReceiver] = {}
        self.camera_status: Dict[str, CameraStatus] = {}
        self.monitoring = False
        self.monitor_thread = None
        
        # 설정값
        self.tolerance = self.config.get('tolerance', 2.0)
        self.check_interval = self.config.get('check_interval', 1.0)
        self.auto_adjust = self.config.get('auto_adjust', True)
        
    def load_config(self) -> dict:
        """설정 파일 로드"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                printf(f"설정 로드 완료: {self.config_file}", ptype=LT.info)
                return config
        except FileNotFoundError:
            printf(f"설정 파일 없음: {self.config_file}, 기본값 사용", ptype=LT.warning)
            return self._get_default_config()
        except Exception as e:
            printf(f"설정 로드 실패: {e}", ptype=LT.error)
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정"""
        return {
            'tolerance': 2.0,
            'check_interval': 1.0,
            'auto_adjust': True,
            'cameras': {}
        }
    
    def save_config(self):
        """현재 설정 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            printf(f"설정 저장 완료: {self.config_file}", ptype=LT.info)
        except Exception as e:
            printf(f"설정 저장 실패: {e}", ptype=LT.error)

    def add_camera(self, camera_id: str, ip_address: str, port: int,
                   reference_orientation: Optional[Dict[str, float]] = None):
        """
        카메라 추가 및 연결
        
        Args:
            camera_id: 카메라 ID
            ip_address: 라즈베리파이 IP
            port: 통신 포트
            reference_orientation: 기준 자세 (None이면 설정에서 로드)
        """
        # 리시버 생성
        receiver = CameraGyroReceiver(camera_id, ip_address, port)
        
        # 연결 시도
        if receiver.connect():
            receiver.start_receiving()
            self.receivers[camera_id] = receiver
            
            # 상태 초기화
            if reference_orientation is None and camera_id in self.config.get('cameras', {}):
                reference_orientation = self.config['cameras'][camera_id].get('reference_orientation')
            
            self.camera_status[camera_id] = CameraStatus(
                camera_id=camera_id,
                ip_address=ip_address,
                port=port,
                connected=True,
                last_update=time.time(),
                reference_orientation=reference_orientation,
                current_orientation=None,
                needs_adjustment=False,
                error=None
            )

            printf(f"[{camera_id}] 카메라 추가 완료", ptype=LT.info)
            return True
        else:
            printf(f"[{camera_id}] 카메라 추가 실패", ptype=LT.error)
            return False
    
    def remove_camera(self, camera_id: str):
        """카메라 제거"""
        if camera_id in self.receivers:
            self.receivers[camera_id].disconnect()
            del self.receivers[camera_id]
            del self.camera_status[camera_id]
            printf(f"[{camera_id}] 카메라 제거", ptype=LT.info)
    
    def check_camera_orientation(self, camera_id: str) -> Optional[dict]:
        """
        카메라의 현재 자세 확인 및 조정 필요 여부 판단
        
        Returns:
            dict: {'needs_adjustment': bool, 'errors': dict, 'current': dict, 'reference': dict}
        """
        if camera_id not in self.receivers or camera_id not in self.camera_status:
            return None
        
        status = self.camera_status[camera_id]
        receiver = self.receivers[camera_id]
        
        # 최신 데이터 가져오기
        latest_data = receiver.get_latest_data()
        
        if latest_data is None:
            status.error = "데이터 없음"
            return None
        
        # 현재 orientation 업데이트
        current = {
            'roll': latest_data.roll,
            'pitch': latest_data.pitch,
            'yaw': latest_data.yaw
        }
        status.current_orientation = current
        status.last_update = latest_data.timestamp
        
        # 기준값이 없으면 현재 값을 기준으로 설정
        if status.reference_orientation is None:
            printf(f"[{camera_id}] 기준 자세 미설정, 현재 자세를 기준으로 설정", ptype= LT.warning)
            status.reference_orientation = current.copy()
            return {'needs_adjustment': False, 'errors': {}, 'current': current, 'reference': current}
        
        # 오차 계산
        reference = status.reference_orientation
        errors = {
            'roll': current['roll'] - reference['roll'],
            'pitch': current['pitch'] - reference['pitch'],
            'yaw': current['yaw'] - reference['yaw']
        }
        
        # 조정 필요 여부
        needs_adjustment = (
            abs(errors['roll']) > self.tolerance or
            abs(errors['pitch']) > self.tolerance
        )
        
        status.needs_adjustment = needs_adjustment
        status.error = None
        
        return {
            'needs_adjustment': needs_adjustment,
            'errors': errors,
            'current': current,
            'reference': reference
        }
    
    def adjust_camera(self, camera_id: str, gain=0.5) -> bool:
        """
        카메라 자세 조정 명령 전송
        
        Args:
            camera_id: 카메라 ID
            gain: 제어 게인 (0.0 ~ 1.0)
        
        Returns:
            bool: 명령 전송 성공 여부
        """
        check_result = self.check_camera_orientation(camera_id)
        
        if check_result is None or not check_result['needs_adjustment']:
            return False
        
        errors = check_result['errors']
        receiver = self.receivers[camera_id]
        
        # 조정 명령 생성
        command = {
            'command': 'adjust_orientation',
            'roll_error': errors['roll'],
            'pitch_error': errors['pitch'],
            'gain': gain,
            'timestamp': time.time()
        }
        
        # 명령 전송
        success = receiver.send_command(command)
        
        if success:
            printf(f"[{camera_id}] 자세 조정 명령 전송: Roll={errors['roll']:+.2f}°, Pitch={errors['pitch']:+.2f}°", ptype=LT.info)
        else:
            printf(f"[{camera_id}] 자세 조정 명령 전송 실패", ptype=LT.error)

        return success
    
    def set_reference_orientation(self, camera_id: str, use_current=True,
                                   orientation: Optional[Dict[str, float]] = None):
        """
        기준 자세 설정
        
        Args:
            camera_id: 카메라 ID
            use_current: True면 현재 자세를 기준으로 설정
            orientation: 직접 지정할 기준 자세
        """
        if camera_id not in self.camera_status:
            printf(f"[{camera_id}] 카메라를 찾을 수 없음", ptype=LT.error)
            return False
        
        status = self.camera_status[camera_id]
        
        if use_current:
            # 현재 자세를 기준으로 설정
            receiver = self.receivers[camera_id]
            latest_data = receiver.get_latest_data()
            
            if latest_data is None:
                printf(f"[{camera_id}] 현재 데이터 없음", ptype=LT.error)
                return False
            
            status.reference_orientation = {
                'roll': latest_data.roll,
                'pitch': latest_data.pitch,
                'yaw': latest_data.yaw
            }
        elif orientation is not None:
            status.reference_orientation = orientation
        else:
            printf(f"[{camera_id}] 기준 자세를 지정하지 않음", ptype=LT.error)
            return False

        printf(f"[{camera_id}] 기준 자세 설정: {status.reference_orientation}", ptype=LT.info)

        # Config에 저장
        if 'cameras' not in self.config:
            self.config['cameras'] = {}
        
        if camera_id not in self.config['cameras']:
            self.config['cameras'][camera_id] = {}
        
        self.config['cameras'][camera_id]['reference_orientation'] = status.reference_orientation
        self.save_config()
        
        return True
    
    def start_monitoring(self):
        """모든 카메라 자동 모니터링 시작"""
        if self.monitoring:
            printf("이미 모니터링 중", ptype=LT.warning)
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        printf("자동 모니터링 시작", ptype=LT.info)
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        printf("자동 모니터링 중지", ptype=LT.info)
    
    def _monitor_loop(self):
        """모니터링 루프 (백그라운드 스레드)"""
        while self.monitoring:
            for camera_id in list(self.receivers.keys()):
                try:
                    check_result = self.check_camera_orientation(camera_id)
                    
                    if check_result and check_result['needs_adjustment']:
                        printf(f"[{camera_id}] 자세 이탈 감지!", ptype=LT.warning)

                        if self.auto_adjust:
                            self.adjust_camera(camera_id, gain=0.3)
                
                except Exception as e:
                    printf(f"[{camera_id}] 모니터링 오류: {e}", ptype=LT.error)
            
            time.sleep(self.check_interval)
    
    def get_all_status(self) -> Dict[str, CameraStatus]:
        """모든 카메라 상태 반환"""
        return self.camera_status.copy()
    
    def print_status(self):
        """모든 카메라 상태 출력"""
        print("\n" + "="*80)
        print("카메라 자이로 시스템 상태")
        print("="*80)
        
        for camera_id, status in self.camera_status.items():
            print(f"\n[{camera_id}]")
            print(f"  주소: {status.ip_address}:{status.port}")
            print(f"  연결: {'✓' if status.connected else '✗'}")
            
            if status.reference_orientation:
                ref = status.reference_orientation
                print(f"  기준: Roll={ref['roll']:.2f}° Pitch={ref['pitch']:.2f}° Yaw={ref['yaw']:.2f}°")
            
            if status.current_orientation:
                cur = status.current_orientation
                print(f"  현재: Roll={cur['roll']:.2f}° Pitch={cur['pitch']:.2f}° Yaw={cur['yaw']:.2f}°")
                
                if status.reference_orientation:
                    err_roll = cur['roll'] - ref['roll']
                    err_pitch = cur['pitch'] - ref['pitch']
                    print(f"  오차: Roll={err_roll:+.2f}° Pitch={err_pitch:+.2f}°")
            
            print(f"  조정 필요: {'예' if status.needs_adjustment else '아니오'}")
            
            if status.error:
                print(f"  오류: {status.error}")
            
            if status.last_update:
                elapsed = time.time() - status.last_update
                print(f"  마지막 업데이트: {elapsed:.1f}초 전")
        
        print("="*80)
    
    def cleanup(self):
        """모든 리소스 정리"""
        printf("시스템 종료 중...", ptype=LT.info)
        
        self.stop_monitoring()
        
        for camera_id in list(self.receivers.keys()):
            self.remove_camera(camera_id)

        printf("시스템 종료 완료", ptype=LT.info)


