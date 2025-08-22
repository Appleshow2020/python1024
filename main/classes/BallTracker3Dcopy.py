import cv2 as cv
import numpy as np
import time

class BallTracker3D:
    """
    공의 위치를 계산하는 클래스

    Functions : detect_ball, triangulate_point, update_state
    initial variables : camera_params
    """
    def __init__(self, camera_params):
        """
        camera_params: dict
            {
                "cam1": {"P": 3x4 projection matrix, "id": 1},
                "cam2": {"P": 3x4 projection matrix, "id": 2},
                ...
            }
        """
        self.camera_params = camera_params
        self.prev_positions = []
        self.prev_times = []

    def detect_ball(self, frame):
        """
        1. HSV 마스킹으로 공 검출 시도
        2. 실패 시 YOLO로 대체 검출
        Return: (u, v) or None
        """
        # HSV 마스킹
        # img=cv.imread(frame)
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_color = np.array([209,108,96])
        upper_color = np.array([115,35,24])
        mask = cv.inRange(hsv, lower_color, upper_color)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 컨투어 중심을 공 위치로 간주
            c = max(contours, key=cv.contourArea)
            M = cv.moments(c)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                return (u, v)

        # # HSV 실패 시 YOLO 사용
        # results = self.model(frame)
        # for result in results:
        #     boxes = result.boxes
        #     for box in boxes:
        #         cls = int(box.cls[0])
        #         conf = float(box.conf[0])
        #         if cls == 0 and conf > 0.5:  # "공" 클래스 번호는 0으로 가정
        #             x1, y1, x2, y2 = map(int, box.xyxy[0])
        #             u = int((x1 + x2) / 2)
        #             v = int((y1 + y2) / 2)
        #             return (u, v)

        return None  # 검출 실패

    def triangulate_point(self, pts_2d, cam_ids):
        """
        pts_2d: list of (u, v) tuples from two cameras
        cam_ids: list of camera IDs corresponding to pts_2d
        Returns: 3D world coordinate (x, y, z)
        """
        assert len(pts_2d) >= 2, f"\033[31m[{time.strftime('%X')}] [ERROR] Not enough camera data for triangulation.\033[0m"

        P1 = self.camera_params[f"cam{cam_ids[0]+1}"]["P"]
        P2 = self.camera_params[f"cam{cam_ids[1]+1}"]["P"]

        pt1 = np.array(pts_2d[0], dtype=np.float32).reshape(2, 1)
        pt2 = np.array(pts_2d[1], dtype=np.float32).reshape(2, 1)

        point_4d = cv.triangulatePoints(P1, P2, pt1, pt2)
        
        if point_4d[3] == 0 or not np.isfinite(point_4d[3]):
            print("\033[31m[ERROR] Invalid triangulation result: point_4d[3] == 0 or NaN\033[0m")
            return np.array([np.nan, np.nan, np.nan])

        point_3d = point_4d[:3] / point_4d[3]
        return point_3d.flatten()
    
    def update_state(self, position_3d, timestamp):
        """
        position_3d: numpy array of shape (3,)
        timestamp: float (time in seconds)
        Returns: dict with position, velocity, acceleration, direction
        """
        self.prev_positions.append(position_3d)
        self.prev_times.append(timestamp)

        if len(self.prev_positions) < 2:
            return {"position": position_3d, "velocity": None, "acceleration": None, "direction": None}

        pos1, pos2 = self.prev_positions[-2], self.prev_positions[-1]
        t1, t2 = self.prev_times[-2], self.prev_times[-1]
        dt = t2 - t1
        if dt == 0:
            return {"position": position_3d, "velocity": None, "acceleration": None, "direction": None}

        velocity = (pos2 - pos1) / dt

        direction = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 0 else None

        return {
            "position": position_3d.tolist() if position_3d is not None else position_3d,
            "velocity": velocity.tolist() if velocity is not None else velocity,
            "direction": direction.tolist() if direction is not None else direction
        }