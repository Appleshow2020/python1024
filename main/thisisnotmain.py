# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import time
import asyncio
import cv2 as cv
from ultralytics import YOLO

from classes.CameraCalibration import CameraCalibration
from classes.Animation import Animation
from classes.UserInterface import UserInterface
from classes.BallTracker3D import BallTracker3D

BALL_SIZE = 0

cv.destroyAllWindows()

class LineCheck:
    def __init__(self,cameranum:int):pass

def KeypointBound(cameranum,dxylist) -> bool:pass

def FloorChecker(height: float) -> bool:
    global Positioned
    if height <= floor+BALL_SIZE:
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
    global Positioned
    if ((bx <= 0)and(bx >= PointList[4][0])) and ((by <= PointList[4][1])and(by>=PointList[8][1])):
        Positioned[4] = True
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = False
        return "li"
    
    elif ((bx >= 0)and(bx <= PointList[7][0])) and ((by <= PointList[7][1])and(by>=PointList[11][1])):
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
        Positioned[4] = False
        Positioned[5] = False
        Positioned[6] = False
        Positioned[7] = False
        return None

def BallOutLinedChecker(x: float, y: float) -> bool: 
    if BallPlaceChecker(x,y) == None:
        Positioned[8] = False
        return True
    Positioned[8] = True
    return False

def RootCheckerRecursion(li: list, i: int, p: int, Flag: bool | None = ValueError) -> any:
    def Local_CheckSequence(start, length) -> bool:
        return all(li[start + k][0] < li[start + k + 1][0] for k in range(length))

    sequence_lengths = {
        0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
    }

    try:
        if p in sequence_lengths:
            length = sequence_lengths[p]
            if Local_CheckSequence(i - length - 1, length):
                return RootCheckerRecursion(li, i - length, p, Flag)
            else:
                return RootCheckerRecursion(li, i, p + 1, Flag)
        elif p == 9:
            if Flag:
                bx = li[i][1]
                by = li[i][2]

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
        return RootCheckerRecursion(li, i, p+1, Flag)

def mod(x: float, y: float) -> float:
    return x%y

def average(x: list) -> float:
    return sum(x)/len(x)

def load_camera_configs_from_txt(filepath):
    camera_configs = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue  # 빈 줄 무시
            parts = line.strip().split()
            if len(parts) != 7:
                raise ValueError(f"Invalid line format: {line}")
            
            cam_id = parts[0]
            pos = list(map(float, parts[1:4]))
            rot = list(map(float, parts[4:7]))
            camera_configs.append({
                "id": cam_id,
                "position": pos,
                "rotation": rot
            })
    return camera_configs

al = {}
gl = {}
pl = {}

default_unpacked = al.values[0] + gl.values[0] + pl.values[0]

default_unpacked[0] /= 16384
default_unpacked[1] /= 16384
default_unpacked[2] /= 16384
default_unpacked[3] /= 131
default_unpacked[4] /= 131
default_unpacked[5] /= 131

floor = default_unpacked[-1]
print(floor)

i = 0

pdx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
pdy = [7, 4, -4, -7]
PointList = []
while (len(PointList)!=16):
    PointList.append([pdx[i], pdy[i//4]])
    i+=1

CameraPointList = []

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out","Running"]
Positioned = [False]*len(Positioned)
responseDict = []
t = 0
tlist= [0]
print(f"[{time.strftime('%X')}] [INFO] Loading camera configurations...",end=" ")
camera_configs = load_camera_configs_from_txt("camera_configs.txt")
print("completed. (7 configs loaded)")
calibrate = CameraCalibration(camera_configs)
calibrate.get_camera_params()
calibrate.print_projection_matrices()

animate = Animation(al, gl, pl)
tracker = BallTracker3D()
interface = UserInterface()
tracker.model = YOLO('yolov8n.pt')
async def main():
    print(f"[{time.strftime('%X')}] [INFO] System successfully initialized.")
    while True:
        timestamp = time.time()

        lal = al.values[-1]
        lgl = gl.values[-1]
        posX, posY, posZ = 0
        accelX, accelY, accelZ, gyroX, gyroY, gyroZ = lal[0], lal[1], lal[2], lgl[0], lgl[1], lgl[2]
        responseDict[timestamp] = (accelX, accelY, accelZ, gyroX, gyroY, gyroZ, posX, posY, posZ)
        al[timestamp] = (accelX,accelY,accelZ)
        gl[timestamp] = (gyroX, gyroY, gyroZ)
        pl[timestamp] = (posX, posY, posZ)
        # try:
        #     if OutLinedChecker(magX,magY):
        #         print('ball outlined')
        #     if FloorChecker(magZ) and HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
        #         continue
        #     if HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
        #         state = RootCheckerRecursion([accelX,accelY,accelZ], 0, 0, True)
        #         if state == True:
        #             print('r player hitted by l player')
        #         elif state == False:
        #             print('l player hitted by r player')
        #         else:
        #             pass
        # except Exception as exception:
        #     print(exception)
        # TODO
            
        elapsed = await time.time()-timestamp
        tlist.append(tlist[-1]+elapsed)
        t+=elapsed
        
asyncio.run(main())
