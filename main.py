# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import sys
import serial as s
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import ExtendedKalmanFilter


def FloorChecker(height): 
    if height == floor:
        Positioned[0] == True
        return True
    else:
        Positioned[0] == False
        return False
def HittedChecker(ax1, ay1, az1, ax2, ay2, az2, mz): 
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

def ThrowerChecker(l,temp): 
    try:
        a = math.sqrt((l[0]**2)+(l[1]**2)+(l[2]**2))
        al.append([a,l[3],l[4],l[5]])
        return ThrowerCheckerRecursion(temp,0)
    except ValueError:
        return None

def ThrowerCheckerRecursion(i,p): 
    try:
        if p==0:
            if all(al[i-j][0] < al[i-j+1][0] for j in range(10, 1, -1)): 
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

def BallPlaceChecker(bx,by): 
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

def OutLinedChecker(x,y): 
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
    
