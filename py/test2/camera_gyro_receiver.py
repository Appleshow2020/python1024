from typing import Optional, List
import socket
import threading
import json
import time
import queue
from utils.printing import printf, LT
from core.models.gyro_data import GyroData

class CameraGyroReceiver:
    def __init__(self, camera_id: str, ip_address: str, port: int):
        self.camera_id = camera_id
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=100)
        self.last_data = None
        
    def connect(self, timeout=5):
        """카메라에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            printf(f"[{self.camera_id}] 연결 성공: {self.ip_address}:{self.port}", ptype=LT.info)
            return True
        except Exception as e:
            printf(f"[{self.camera_id}] 연결 실패: {e}", ptype=LT.error)
            self.connected = False
            return False
    
    def disconnect(self):
        """연결 종료"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        printf(f"[{self.camera_id}] 연결 종료", ptype=LT.info)

    def start_receiving(self):
        """데이터 수신 시작 (백그라운드 스레드)"""
        if not self.connected:
            printf(f"[{self.camera_id}] 연결되지 않음", ptype=LT.warning)
            return False
        
        self.running = True
        self.receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True
        )
        self.receive_thread.start()
        printf(f"[{self.camera_id}] 데이터 수신 시작", ptype=LT.info)
        return True
    
    def _receive_loop(self):
        """데이터 수신 루프"""
        buffer = ""
        
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                
                if not data:
                    printf(f"[{self.camera_id}] 연결 끊김", ptype=LT.warning)
                    self.connected = False
                    break
                
                buffer += data
                
                # JSON 객체 파싱 (줄바꿈으로 구분)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    try:
                        json_data = json.loads(line)
                        gyro_data = GyroData(
                            camera_id=self.camera_id,
                            timestamp=json_data.get('timestamp', time.time()),
                            roll=json_data['roll'],
                            pitch=json_data['pitch'],
                            yaw=json_data['yaw'],
                            gyro_x=json_data.get('gyro_x', 0),
                            gyro_y=json_data.get('gyro_y', 0),
                            gyro_z=json_data.get('gyro_z', 0),
                            motor_pan=json_data.get('motor_pan', 90),
                            motor_tilt=json_data.get('motor_tilt', 90)
                        )
                        
                        # 큐에 데이터 추가
                        try:
                            self.data_queue.put_nowait(gyro_data)
                            self.last_data = gyro_data
                        except queue.Full:
                            # 큐가 가득 차면 가장 오래된 데이터 제거
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(gyro_data)
                            except:
                                pass
                    
                    except json.JSONDecodeError as e:
                        printf(f"[{self.camera_id}] JSON 파싱 오류: {e}", ptype=LT.debug)
                        continue
                
            except socket.timeout:
                continue
            except Exception as e:
                printf(f"[{self.camera_id}] 수신 오류: {e}", ptype=LT.error)
                self.connected = False
                break
    
    def get_latest_data(self) -> Optional[GyroData]:
        """최신 데이터 가져오기"""
        return self.last_data
    
    def get_data_batch(self, max_count=10) -> List[GyroData]:
        """큐에서 여러 데이터 가져오기"""
        batch = []
        while not self.data_queue.empty() and len(batch) < max_count:
            try:
                batch.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return batch
    
    def send_command(self, command: dict) -> bool:
        """카메라에 명령 전송 (자세 조정 등)"""
        if not self.connected:
            return False
        
        try:
            message = json.dumps(command) + '\n'
            self.socket.sendall(message.encode('utf-8'))
            printf(f"[{self.camera_id}] 명령 전송: {command}", ptype=LT.debug)
            return True
        except Exception as e:
            printf(f"[{self.camera_id}] 명령 전송 실패: {e}", ptype=LT.error)
            return False