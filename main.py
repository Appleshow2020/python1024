# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import sys
import serial as s
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(0)

class ExtendedKalmanFilter():
    def get_radar(self, xpos_pred):
        """레이다로 측정된 고도와 직선거리를 반환해줌"""
        xvel_w = np.random.normal(0, 5)   # xvel_w: 이동거리의 시스템 잡음 [m/s].
        xvel_true = 100 + xvel_w          # xvel_true: 이동거리의 참값 [m/s].
    
        ypos_w = np.random.normal(0, 10)  # ypos_w: 고도의 시스템 잡음 [m].
        ypos_true = 1000 + ypos_w         # ypos_true: 고도의 참값 [m].
    
        xpos_pred = xpos_pred + xvel_true * self.dt                     # xpos_pred: 이동거리 예상치 [m].
    
        rpos_v = xpos_pred * np.random.normal(0, 0.05)             # rpos_v: 레이다의 측정잡음.
        rpos_meas = np.sqrt(xpos_pred**2 + ypos_true**2) + rpos_v  # r: 측정 거리 [m] (observable).
    
        return rpos_meas, xpos_pred
    # 야코비안 계산
    def Ajacob_at(self, x_esti):
        return self.A
    
    def Hjacob_at(self, x_pred):
        self.H[0][0] = x_pred[0] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
        self.H[0][1] = 0
        self.H[0][2] = x_pred[2] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
        return self.H


    # 비선형시스템 계산 (측정모델)
    def fx(self, x_esti):
        return self.A @ x_esti
    
    def hx(self, x_pred):
        self.z_pred = np.sqrt(self.x_pred[0]**2 + self.x_pred[2]**2)
        return np.array([self.z_pred])
    def extended_kalman_filter(self, z_meas, x_esti, P):
        """Extended Kalman Filter Algorithm."""
        # (1) Prediction.
        A = self.Ajacob_at(x_esti)
        x_pred = self.fx(x_esti)
        P_pred = A @ P @ A.T + self.Q
    
        # (2) Kalman Gain.
        H = self.jacob_at(x_pred)
        K = P_pred @ H.T @ inv(H @ P_pred @ H.T + self.R)
    
        # (3) Estimation.
        x_esti = x_pred + K @ (z_meas - self.hx(x_pred))
    
        # (4) Error Covariance.
        P = P_pred - K @ H @ P_pred
    
        return x_esti, P
    def __init__(self):
        # Input parameters.
        time_end = 20
        dt = 0.05


        # Initialization for system model.
        # Matrix: A, H, Q, R, P_0
        # Vector: x_0
        self.A = np.eye(3) + dt * np.array([[0, 1, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]])
        self.H = np.zeros((1, 3))
        self.Q = np.array([[0, 0, 0],
                      [0, 0.001, 0],
                      [0, 0, 0.001]])
        self.R = np.array([[10]])

        # Initialization for estimation.
        self.x_0 = np.array([0, 90, 1100])  # [horizontal position, horizontal velocity, vertical position].
        self.P_0 = 10 * np.eye(3)


        self.time = np.arange(0, time_end, dt)
        self.n_samples = len(self.time)
        self.xpos_esti_save = np.zeros(self.n_samples)
        self.ypos_esti_save = np.zeros(self.n_samples)
        self.rpos_esti_save = np.zeros(self.n_samples)
        self.xvel_esti_save = np.zeros(self.n_samples)
        self.rpos_meas_save = np.zeros(self.n_samples)
        self.xpos_pred = 0
        self.x_esti, P = None, None
        for i in range(self.n_samples):
            z_meas, xpos_pred = self.get_radar(xpos_pred)
            if i == 0:
                x_esti, P = self.x_0, self.P_0
            else:
                x_esti, P = self.extended_kalman_filter(z_meas, x_esti, P)

            self.xpos_esti_save[i] = x_esti[0]
            self.ypos_esti_save[i] = x_esti[2]
            self.rpos_esti_save[i] = np.sqrt(x_esti[0]**2 + x_esti[2]**2)
            self.xvel_esti_save[i] = x_esti[1]
            self.rpos_meas_save[i] = z_meas
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        axes[0, 0].plot(self.time, self.xpos_esti_save, 'bo-', label='Estimation (EKF)')
        axes[0, 0].legend(loc='upper left')
        axes[0, 0].set_title('Horizontal Distance: Esti. (EKF)')
        axes[0, 0].set_xlabel('Time [sec]')
        axes[0, 0].set_ylabel('Horizontal Distance [m]')

        axes[0, 1].plot(self.time, self.ypos_esti_save, 'bo-', label='Estimation (EKF)')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].set_title('Vertical Distance: Esti. (EKF)')
        axes[0, 1].set_xlabel('Time [sec]')
        axes[0, 1].set_ylabel('Vertical Distance [m]')

        axes[1, 0].plot(self.time, self.rpos_meas_save, 'r*--', label='Measurements', markersize=10)
        axes[1, 0].plot(self.time, self.rpos_esti_save, 'bo-', label='Estimation (EKF)')
        axes[1, 0].legend(loc='upper left')
        axes[1, 0].set_title('Radar Distance: Meas. v.s. Esti. (EKF)')
        axes[1, 0].set_xlabel('Time [sec]')
        axes[1, 0].set_ylabel('Radar Distance [m]')

        axes[1, 1].plot(self.time, self.xvel_esti_save, 'bo-', label='Estimation (EKF)')
        axes[1, 1].legend(loc='upper left')
        axes[1, 1].set_title('Horizontal Velocity: Esti. (EKF)')
        axes[1, 1].set_xlabel('Time [sec]')
        axes[1, 1].set_ylabel('Horizontal Velocity [m/s]')

def FloorChecker(height): # 바닥에 있는가를 체크하는 함수 ( O(1) )
    if height == floor:
        Positioned[0] == True
        return True
    else:
        Positioned[0] == False
        return False
def HittedChecker(ax1, ay1, az1, ax2, ay2, az2, mz): # 맞았는가를 체크하는 함수 ( O(1) )
    try:
        if FloorChecker(mz):
            return False
            
        if (ax1<ax2) and (ay1<ay2) and not (az1<az2):
            Positioned[1] == True
        else:
            Positioned[1] == False
    except ValueError:
        return False
    return True

def ThrowerChecker(l,temp): # 던진 사람을 체크할때 사용되는 리스트 추가 함수 ( O(1) )
    try:
        a = math.sqrt((l[0]**2)+(l[1]**2)+(l[2]**2))
        al.append([a,l[3],l[4],l[5]])
        return ThrowerCheckerRecursion(temp,0)
    except ValueError:
        return None

def ThrowerCheckerRecursion(i,p): # 던진 사람을 재귀적으로 체크할때 사용되는 메인 함수 ( O(n) )
    try:
        if p==0:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(10, 1, -1)): # == if al[i-10][0]<al[i-9][0]<al[i-8][0]<al[i-7][0]<al[i-6][0]<al[i-5][0]<al[i-4][0]<al[i-3][0]<al[i-2][0]<al[i-1][0]:
                return ThrowerCheckerRecursion(i-9,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==1:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(9, 1, -1)):
                return ThrowerCheckerRecursion(i-8,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==2:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(8, 1, -1)):
                return ThrowerCheckerRecursion(i-7,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==3:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(7, 1, -1)):
                return ThrowerCheckerRecursion(i-6,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==4:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(6, 1, -1)):
                return ThrowerCheckerRecursion(i-5,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==5:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(5, 1, -1)):
                return ThrowerCheckerRecursion(i-4,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==6:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(4, 1, -1)):
                return ThrowerCheckerRecursion(i-3,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==7:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(3, 1, -1)):
                return ThrowerCheckerRecursion(i-2,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==8:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(2, 1, -1)):
                return ThrowerCheckerRecursion(i-1,p)
            else:
                return ThrowerCheckerRecursion(i,p+1)
        elif p==9:

            bx = al[i][1]
            by = al[i][2]

            if (BallPlaceChecker(bx,by) == "li") or (BallPlaceChecker(bx,by) == "lo"):
                return True
            elif (BallPlaceChecker(bx,by) == "ri") or (BallPlaceChecker(bx,by) == "ro"):
                return False
            else:
                return None
        else:
            return ValueError

    except IndexError:
        return ThrowerCheckerRecursion(i,p+1)

def BallPlaceChecker(bx,by): # 공의 위치를 확인하는 함수 ( O(1) )
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

def OutLinedChecker(x,y): # 경기장 밖으로 나갔는지 체크하는 함수 ( O(1) )
    if BallPlaceChecker(x,y) == None:
        Positioned[3] == True
        return True
    Positioned[3] == False
    return False

input = sys.stdin.readline

Port=int(input())
ser = s.Serial("COM{}".format(Port),baudrate=115200)
a = int(input())
x, y, floor = map(int, input().split())

i = 0

dx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
dy = [7, 4, -4, -7]
PointList = []

while (len(PointList)!=len(dx)):
    PointList.append([(a/180)*dx[i] if a!=0 else dx[i], (a/180)*dy[i//4] if a!=0 else dy[i//4]])
    i+=1
print(*PointList)
ser.write(bytes(f'Connected as Port: {ser.portstr}, baudrate:{ser.baudrate}', encoding='ascii'))

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "R Out", "L Out"]
Positioned = [False]*len(Positioned)
responseList = []
al = []
while True:
    kalman = ExtendedKalmanFilter

    response = ser.readline().decode()
    responseList.append(list(response.split(',')))
    accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ = response.split(',')
    FloorChecker(magZ)
    HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ)
    ThrowerChecker(accelX,accelY,accelZ,0)
    OutLinedChecker(magX,magY)
    
