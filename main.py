# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import os
import serial
import copy
import time
import argparse
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import kagglehub
import ctypes
import struct

cv.destroyAllWindows()

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, x):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Estimate error covariance
        self.x = x  # State estimate

    def predict(self, u=np.zeros(1)):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x

class Position:
    def __init__(self):
        self.kf = self.KalmanFilterInitialize()
    def KalmanFilterInitialize(self):
        A = np.eye(6)
        B = np.eye(6)
        H = np.zeros((3,6))
        H[:3, :3] = np.eye(3)
        Q = np.eye(6) * 0.1
        R = np.eye(3) * 0.1
        P = np.eye(6)
        x = np.zeros(6)
        return KalmanFilter(A, B, H, Q, R, P, x)
    def OrientationUpdate(self, orientation, gyro_data, dt):
        self.omega = np.array([0, gyro_data[0], gyro_data[1], gyro_data[2]])
        self.dq = 0.5 * self.MultiplyQuaternion(orientation, self.omega)
        return orientation + self.dq * dt

    def MultiplyQuaternion(self, q, r):
        self.w1, self.x1, self.y1, self.z1 = q
        self.w2, self.x2, self.y2, self.z2 = r
        return np.array([
            self.w1 * self.w2 - self.x1 * self.x2 - self.y1 * self.y2 - self.z1 * self.z2,
            self.w1 * self.x2 + self.x1 * self.w2 + self.y1 * self.z2 - self.z1 * self.y2,
            self.w1 * self.y2 - self.x1 * self.z2 + self.y1 * self.w2 + self.z1 * self.x2,
            self.w1 * self.z2 + self.x1 * self.y2 - self.y1 * self.x2 + self.z1 * self.w2
        ])

    def VectorRotate(self, vector, quaternion):
        self.q_conjugate = quaternion * np.array([1, -1, -1, -1])
        self.q_vector = np.concatenate(([0], vector))
        self.rotated_vector = self.MultiplyQuaternion(self.MultiplyQuaternion(quaternion, self.q_vector), self.q_conjugate)
        return self.rotated_vector[1:]


    def PositionCompute(self, acc_data, gyro_data, initial_position, initial_velocity, dt):
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)
        self.orientation = np.array([1, 0, 0, 0])

        for acc, gyro in zip(acc_data, gyro_data):
            self.orientation = self.OrientationUpdate(self.orientation, gyro, dt)
            self.acc_world = self.VectorRotate(acc, self.orientation)
            #self.velocity += self.acc_world * dt
            self.velocity = np.add(self.velocity, np.multiply(self.acc_world, dt), out=self.velocity, casting='unsafe')
            self.position += self.velocity * dt

            z = self.position[:3]
            updated_state = self.kf.update(z)
            self.position[:3] = updated_state[:3]
        return self.position


class Multipose:
    def __init__(self):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    def InitializingMedia(self):
        MFStartup = ctypes.windll.mfplat.MFStartup
        MFShutdown = ctypes.windll.mfplat.MFShutdown
        MF_VERSION = 0x00020070

        hr = MFStartup(MF_VERSION, 0)
        if hr != 0:
            raise Exception('hr!=0')
        return MFShutdown
    
    def Arguments(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--device", type=int, default=0)
        self.parser.add_argument("--file", type=str, default=None)
        self.parser.add_argument("--width", help='cap width', type=int, default=960)
        self.parser.add_argument("--height", help='cap height', type=int, default=540)
        self.parser.add_argument('--mirror', action='store_true', default=True)
        self.parser.add_argument("--keypoint_score", type=float, default=0.4)
        self.parser.add_argument("--bbox_score", type=float, default=0.3)
        self.args = self.parser.parse_args()
        return self.args
 
    def Inference(self, model, input_size, image):
        self.image_width, self.image_height = image.shape[1], image.shape[0]
        self.input_image = cv.resize(image, dsize=(input_size, input_size))  
        self.input_image = cv.cvtColor(self.input_image, cv.COLOR_BGR2RGB)  
        self.input_image = self.input_image.reshape(-1, input_size, input_size, 3)  
        self.input_image = tf.cast(self.input_image, dtype=tf.int32)  
        self.outputs = model(self.input_image)
        self.keypoints_with_scores = self.outputs['output_0'].numpy()
        self.keypoints_with_scores = np.squeeze(self.keypoints_with_scores)
        self.keypoints_list, self.scores_list = [], []
        self.bbox_list = []
        for i in self.keypoints_with_scores:
            self.keypoints = []
            self.scores = []
            for index in range(17):
                self.keypoint_x = int(self.image_width * i[(index * 3) + 1])
                self.keypoint_y = int(self.image_height * i[(index * 3) + 0])
                self.score = i[(index * 3) + 2]
                self.keypoints.append([self.keypoint_x, self.keypoint_y])
                self.scores.append(self.score)
            self.bbox_ymin = int(self.image_height * i[51])
            self.bbox_xmin = int(self.image_width * i[52])
            self.bbox_ymax = int(self.image_height * i[53])
            self.bbox_xmax = int(self.image_width * i[54])
            self.bbox_score = i[55]
            self.keypoints_list.append(self.keypoints)
            self.scores_list.append(self.scores)
            self.bbox_list.append(
                [self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax, self.bbox_score])
        return self.keypoints_list, self.scores_list, self.bbox_list

    def main(self):
        self.args = self.Arguments()
        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height
        self.MFShutdown = self.InitializingMedia()
        if self.args.file is not None:
            self.cap_device = self.args.file
        self.mirror = self.args.mirror
        self.keypoint_score_th = self.args.keypoint_score
        self.bbox_score_th = self.args.bbox_score
        self.cap = cv.VideoCapture(0)
        self.cap.open(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.model_url = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
        self.input_size = 256
        self.module = tfhub.load(self.model_url)
        self.model = self.module.signatures['serving_default']
        self.temp = 0
        while True:
            self.start_time = time.time()
            self.ret, self.frame = self.cap.read()
            print(self.ret)
            if not self.ret:
                if self.temp > 3:
                    break
                print(f't{self.temp}')
                time.sleep(1)
                self.temp += 1
                continue
            if self.mirror:
                self.frame = cv.flip(self.frame, 1)  
            self.debug_image = copy.deepcopy(self.frame)
            keypoints_list, scores_list, bbox_list = self.Inference(self.model, self.input_size, self.frame)
            self.elapsed_time = time.time() - self.start_time
            self.debug_image = self.DebugDraw(
                self.debug_image,
                self.elapsed_time,
                self.keypoint_score_th,
                keypoints_list,
                scores_list,
                self.bbox_score_th,
                bbox_list
            )
            self.key = cv.waitKey(1)
            if self.key == 27:  
                break
            cv.imshow('cam1', self.debug_image)
        self.cap.release()
        cv.destroyAllWindows()
        self.MFShutdown()

    def DebugDraw(self, image, elapsed_time, keypoint_score_th, keypoints_list, scores_list, bbox_score_th, bbox_list):
        self.debug_image = copy.deepcopy(image)
        for idx1, idx2 in [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(11,12),(5,11),(11,13),(13,15),(6,12),(12,14),(14,16)]:
            if self.scores[idx1] > keypoint_score_th and self.scores[idx2] > keypoint_score_th:
                self.point01 = self.keypoints[idx1]
                self.point02 = self.keypoints[idx2]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)
        for keypoint, score in zip(keypoints_list, scores_list):
                if score > self.keypoint_score_th:
                    cv.circle(self.debug_image, keypoint, 6, (255, 255, 255), -1)
                    cv.circle(self.debug_image, keypoint, 3, (0, 0, 0), -1)
                    if KeypointBound(keypoint):
                        cv.putText(self.debug_image, 'Keypoint out of bounds', keypoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
        for bbox in bbox_list:
            if bbox[4] > bbox_score_th:
                cv.rectangle(self.debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 4)
                cv.rectangle(self.debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), 2)
        cv.putText(self.debug_image, "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv.LINE_AA)
        cv.putText(self.debug_image, "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv.LINE_AA)
        return self.debug_image
    
    
def KeypointBound(keypoint: int) -> bool:
    x, y = keypoint
    return x<0

def FloorChecker(height: float) -> bool:
    if height == floor:
        Positioned[0] = True
        return True
    else:
        Positioned[0] = False
        return False
def HittedChecker(ax1: float, ay1: float, az1: float,
                  ax2: float, ay2: float, az2: float,
                  mz: float) -> bool: 
    try:
        if FloorChecker(mz):
            return False
            
        if (ax1<ax2) and (ay1<ay2) and not (az1<az2):
            Positioned[1] = True
        else:
            Positioned[1] = False
    except ValueError:
        return False
    return True

def BallPlaceChecker(bx: float, by: float) -> str: 
    if ((bx <= x)and(bx >= PointList[4][0])) and ((by <= PointList[4][1])and(by>=PointList[8][1])):
        Positioned[4] = True
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = False
        return "li"
    
    elif ((bx >= x)and(bx <= PointList[7][0])) and ((by <= PointList[7][1])and(by>=PointList[11][1])):
        Positioned[4] = False
        Positioned[5] = True
        Positioned[6] = False
        Positioned[7] = False
        return "ri"
    
    elif ((((bx >= PointList[2][0])and(bx <= PointList[3][0])) and ((by <= PointList[2][1])and(by >= PointList[6][1]))) or 
          (((bx >= PointList[7][0])and(bx <= PointList[3][0])) and ((by <= PointList[7][1])and(by >= PointList[11][1]))) or
          (((bx >= PointList[10][0])and(bx <= PointList[15][0])) and ((by <= PointList[10][1])and(by >= PointList[14][1])))):
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = True
        Positioned[7] = False
        return "lo"
    
    elif ((((bx >= PointList[0][0])and(bx <= PointList[1][0])) and ((by <= PointList[0][1])and(by >= PointList[5][1]))) or 
          (((bx >= PointList[0][0])and(bx <= PointList[4][0])) and ((by <= PointList[4][1])and(by >= PointList[8][1]))) or
          (((bx >= PointList[12][0])and(bx <= PointList[13][0])) and ((by <= PointList[8][1])and(by >= PointList[13][1])))):
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = True
        return "ro"
    
    else:
        return None

def OutLinedChecker(x: float, y: float) -> bool: 
    if BallPlaceChecker(x,y) == None:
        Positioned[3] = True
        return True
    Positioned[3] = False
    return False

def RootCheckerRecursion(li: list, i: int, p: int, Flag: bool) -> any:
    def check_sequence(start, length):
        return all(li[start + k][0] < li[start + k + 1][0] for k in range(length))

    sequence_lengths = {
        0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
    }

    try:
        if p in sequence_lengths:
            length = sequence_lengths[p]
            if check_sequence(i - length - 1, length):
                return RootCheckerRecursion(li, i - length, p, Flag)
            else:
                return RootCheckerRecursion(li, i, p + 1, Flag)
        elif p == 9:
            if Flag:
                bx = al[i][1]
                by = al[i][2]

                if (BallPlaceChecker(bx, by) == "li") or (BallPlaceChecker(bx, by) == "lo"):
                    return True
                elif (BallPlaceChecker(bx, by) == "ri") or (BallPlaceChecker(bx, by) == "ro"):
                    return False
                else:
                    return None
            else:
                return li[i]
        else:
            raise Exception()
    except IndexError:
        RootCheckerRecursion(li, i, p+1, Flag)

def sqrt(x: float) -> float:
    return x**0.5

def mod(x: float, y: float) -> float:
    return x%y

def S2D(scientific_str: any) -> any:
    decimal_value = float(scientific_str)
    return format(decimal_value, 'f')


print('a')
Port=int(input())

try:
    ser = serial.Serial("COM{}".format(Port), 2000000)
except Exception as e:
    print(e)
# a = float(input())
# x, y, floor = map(float, input().split())

default = ser.read(36)
default_unpacked = struct.unpack('<9f',default)
default_unpacked = list(default_unpacked)
default_unpacked[0] /= 16384
default_unpacked[1] /= 16384
default_unpacked[2] /= 16384
default_unpacked[3] /= 131
default_unpacked[4] /= 131
default_unpacked[5] /= 131
default_unpacked = tuple(default_unpacked)
a = mod(((default_unpacked[3]+default_unpacked[4])/2), 360)
x, y, floor = default_unpacked[-3], default_unpacked[-2], default_unpacked[-1]
print(a, x, y, floor,sep='\n')

i = 0


dx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
dy = [7, 4, -4, -7]
PointList = []
while (len(PointList)!=16):
    PointList.append([(a/180)*dx[i] if a!=0 else dx[i], (a/180)*dy[i//4] if a!=0 else dy[i//4]])
    i+=1
print(*PointList)

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out"]
Positioned = [False]*len(Positioned)
responseList = []
al = [[default_unpacked[0],default_unpacked[1],default_unpacked[2]]]
gl = [[default_unpacked[3],default_unpacked[4],default_unpacked[5]]]
t = 0
while True:
    st = time.time()
    response = ser.read(36)
    unpacked = struct.unpack('<9f', response)
    unpacked = list(unpacked)
    unpacked[0] /= 16384
    unpacked[1] /= 16384
    unpacked[2] /= 16384
    unpacked[3] /= 131
    unpacked[4] /= 131
    unpacked[5] /= 131
    unpacked = tuple(unpacked)
    responseList.append(unpacked)
    accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ = unpacked[0],unpacked[1],unpacked[2],unpacked[3],unpacked[4],unpacked[5],unpacked[6],unpacked[7],unpacked[8]
    calcPose = Position()
    calcPose.PositionCompute(al, gl, [x, y, floor], [0, 0, 0], 0.001)
    print(responseList[-1])
    try:
        if OutLinedChecker(magX,magY):
            ser.write('ball outlined\n')
            print('ball outlined')
        if FloorChecker(magZ) and HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
            continue
        if HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
            state = RootCheckerRecursion([accelX,accelY,accelZ], 0, 0, True)
            if state == True:
                ser.write('r player hitted by l player\n')
                print('r player hitted by l player')
            elif state == False:
                ser.write('l player hitted by r player\n')
                print('l player hitted by r player')
            else:
                raise Exception()
        pose = Multipose()
        pose.main()
    except:
        continue 
    
    t += time.time()-st
    print(t)
