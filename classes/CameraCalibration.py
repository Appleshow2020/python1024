import numpy as np
import time
from classes.printing import *

class CameraCalibration:
    """
    A class for handling camera calibration and projection matrix generation.
    This class manages camera parameters and generates projection matrices for multiple cameras
    based on their position and orientation in 3D space.
    Attributes:
        camera_configs (list): List of dictionaries containing camera configurations.
        K (numpy.ndarray): 3x3 camera intrinsic matrix.
        camera_params (dict): Dictionary containing generated camera parameters.
    Methods:
        _default_intrinsic_matrix(fx, fy, cx, cy): Creates default intrinsic matrix.
        _rotation_matrix_from_euler(pitch, yaw, roll): Converts Euler angles to rotation matrix.
        _create_projection_matrix(R, t): Creates projection matrix from rotation and translation.
        _generate_camera_params(): Generates camera parameters for all configured cameras.
        get_camera_params(): Returns camera parameters.
        print_projection_matrices(): Prints projection matrices for all cameras.
    Example:
        camera_configs = [
                "position": [0, 0, 0],
                "rotation": [0, 0, 0]
        ]
        calib = CameraCalibration(camera_configs, fx, fy, cx, cy)
    """
    
    def __init__(self, camera_configs,*args):
        fx,fy,cx,cy = args
        self.camera_configs = camera_configs
        self.K = self._default_intrinsic_matrix(fx,fy,cx,cy)
        printf(f"Generating camera parameters...", ptype=LT.info)
        self.camera_params = self._generate_camera_params()
        print("completed. (7 parameters generated.)")
    def _default_intrinsic_matrix(self,fx,fy,cx,cy):
        return np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])

    def _rotation_matrix_from_euler(self, pitch, yaw, roll):
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [     0,       1,     0      ],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll),  np.cos(roll), 0],
            [     0,              0,      1]
        ])

        return Rz @ Ry @ Rx

    def _create_projection_matrix(self, R, t):
        Rt = np.hstack((R, t.reshape(3, 1)))  # [R | t]
        return self.K @ Rt  # P = K * [R | t]

    def _generate_camera_params(self):
        camera_params = {}
        for cam in self.camera_configs:
            cam_id = cam["id"]
            position = np.array(cam["position"])
            rotation = self._rotation_matrix_from_euler(*cam["rotation"])
            t = -rotation @ position  # world → camera 기준

            P = self._create_projection_matrix(rotation, t)
            camera_params[cam_id] = {
                "P": P,
                "id": cam_id
            }
        return camera_params

    def get_camera_params(self):
        return self.camera_params

    def print_projection_matrices(self):
        printf(ptype=LT.info,end=" ",useReset=False)
        for cam_id, data in self.camera_params.items():
            print(f"{cam_id} Projection Matrix:\n{data['P']}")
        print(Colors.reset)