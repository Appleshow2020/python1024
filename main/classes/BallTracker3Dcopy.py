import cv2 as cv
import numpy as np
import time

class BallTracker3D:
    """
    개선된 공 위치 계산 클래스
    """
    def __init__(self, camera_params):
        self.camera_params = camera_params
        self.prev_positions = []
        self.prev_times = []
        
        # HSV 색상 범위 개선 (더 넓은 범위)
        self.lower_colors = [
            (0, 30, 30),      # 빨간색 계열 1
            (160, 30, 30),    # 빨간색 계열 2  
            (10, 50, 50),     # 주황색 계열
        ]
        self.upper_colors = [
            (15, 255, 255),   # 빨간색 계열 1
            (179, 255, 255),  # 빨간색 계열 2
            (25, 255, 255),   # 주황색 계열
        ]
        
        # 검출 필터 설정
        self.min_area = 50        # 최소 컨투어 면적
        self.max_area = 5000      # 최대 컨투어 면적
        self.min_circularity = 0.3 # 최소 원형도

    def detect_ball(self, frame):
        """
        개선된 볼 검출 알고리즘
        1. 여러 HSV 범위로 검출 시도
        2. 형태학적 연산으로 노이즈 제거
        3. 원형도 검사로 공 같은 객체 필터링
        """
        if frame is None:
            return None
            
        # 프레임 전처리
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        # 여러 색상 범위로 마스크 생성
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in zip(self.lower_colors, self.upper_colors):
            mask = cv.inRange(hsv, lower, upper)
            combined_mask = cv.bitwise_or(combined_mask, mask)
        
        # 형태학적 연산으로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        
        # 컨투어 찾기
        contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 최적의 컨투어 찾기
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv.contourArea(contour)
            
            # 면적 필터
            if area < self.min_area or area > self.max_area:
                continue
                
            # 원형도 계산
            perimeter = cv.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 원형도 필터
            if circularity < self.min_circularity:
                continue
                
            # 점수 계산 (면적과 원형도 조합)
            score = area * circularity
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is not None:
            # 중심점 계산
            M = cv.moments(best_contour)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                return (u, v)
        
        return None

    def triangulate_point(self, pts_2d, cam_ids):
        """
        개선된 삼각측량 함수
        """
        if len(pts_2d) < 2:
            print(f"\033[31m[{time.strftime('%X')}] [ERROR] Not enough camera data for triangulation.\033[0m")
            return np.array([np.nan, np.nan, np.nan])
        
        try:
            # 카메라 파라미터 가져오기 (인덱스 보정)
            cam1_key = f"cam{cam_ids[0] + 1}"
            cam2_key = f"cam{cam_ids[1] + 1}"
            
            if cam1_key not in self.camera_params or cam2_key not in self.camera_params:
                print(f"\033[31m[ERROR] Camera parameters not found for {cam1_key} or {cam2_key}\033[0m")
                print(f"Available cameras: {list(self.camera_params.keys())}")
                return np.array([np.nan, np.nan, np.nan])
            
            P1 = self.camera_params[cam1_key]["P"]
            P2 = self.camera_params[cam2_key]["P"]
            
            # 2D 포인트 준비
            pt1 = np.array(pts_2d[0], dtype=np.float32).reshape(2, 1)
            pt2 = np.array(pts_2d[1], dtype=np.float32).reshape(2, 1)
            
            # 삼각측량 수행
            point_4d = cv.triangulatePoints(P1, P2, pt1, pt2)
            
            # 결과 검증
            if abs(point_4d[3, 0]) < 1e-10:
                print("\033[31m[ERROR] Triangulation failed: homogeneous coordinate is too small\033[0m")
                return np.array([np.nan, np.nan, np.nan])
            
            # 3D 포인트 계산
            point_3d = point_4d[:3, 0] / point_4d[3, 0]
            
            # 결과 유효성 검사
            if not np.all(np.isfinite(point_3d)):
                print("\033[31m[ERROR] Triangulation result contains NaN or infinity\033[0m")
                return np.array([np.nan, np.nan, np.nan])
            
            # 합리적인 범위 검사 (예: -100m ~ 100m)
            if np.any(np.abs(point_3d) > 100):
                print(f"\033[33m[WARNING] Triangulation result seems unrealistic: {point_3d}\033[0m")
            
            return point_3d
            
        except Exception as e:
            print(f"\033[31m[ERROR] Triangulation exception: {e}\033[0m")
            return np.array([np.nan, np.nan, np.nan])
    
    def update_state(self, position_3d, timestamp):
        """
        상태 업데이트 (속도, 가속도, 방향 계산)
        """
        if np.any(np.isnan(position_3d)):
            return {
                "position": [np.nan, np.nan, np.nan],
                "velocity": [np.nan, np.nan, np.nan],
                "direction": [np.nan, np.nan, np.nan]
            }
        
        self.prev_positions.append(position_3d)
        self.prev_times.append(timestamp)
        
        # 최대 10개의 이전 위치만 보관
        if len(self.prev_positions) > 10:
            self.prev_positions.pop(0)
            self.prev_times.pop(0)
        
        if len(self.prev_positions) < 2:
            return {
                "position": position_3d.tolist(),
                "velocity": [0, 0, 0],
                "direction": [0, 0, 0]
            }
        
        # 속도 계산 (최근 2개 위치 사용)
        pos1, pos2 = self.prev_positions[-2], self.prev_positions[-1]
        t1, t2 = self.prev_times[-2], self.prev_times[-1]
        dt = t2 - t1
        
        if dt <= 0:
            velocity = np.array([0, 0, 0])
        else:
            velocity = (pos2 - pos1) / dt
        
        # 방향 계산
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > 0:
            direction = velocity / velocity_magnitude
        else:
            direction = np.array([0, 0, 0])
        
        return {
            "position": position_3d.tolist(),
            "velocity": velocity.tolist(),
            "direction": direction.tolist()
        }