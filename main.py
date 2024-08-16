# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import os
# import serial
import copy
import time
import argparse
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import kagglehub
import ctypes
# import struct
# import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import asyncio
from bleak import BleakScanner, BleakClient

SERVICE_UUID = "f5d1f9c8-c2dd-4632-a9db-9568a01847ab"
ACCEL_CHARACTERISTIC_UUID = "9c352853-d553-48b2-b192-df074b94bc92"
GYRO_CHARACTERISTIC_UUID = "9c352853-d553-48b2-b192-df074b94bc93"
MAG_CHARACTERISTIC_UUID = "9c352853-d553-48b2-b192-df074b94bc94"

cv.destroyAllWindows()

class Position:
    def __init__(self, initial_position, initial_velocity, dt):
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity, dtype=np.float64)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.dt = dt

    def OrientationUpdate(self, gyro_data):
        gyro_data = np.array(gyro_data)
        omega = np.array([0.0, gyro_data[0], gyro_data[1], gyro_data[2]])
        dq = 0.5 * self.MultiplyQuaternion(self.orientation, omega) * self.dt
        self.orientation += dq

    def MultiplyQuaternion(self, q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def RotateVector(self, vector):
        q_conjugate = self.orientation * np.array([1, -1, -1, -1])
        q_vector = np.concatenate(([0], vector))
        rotated_vector = self.MultiplyQuaternion(self.MultiplyQuaternion(self.orientation, q_vector), q_conjugate)
        return rotated_vector[1:]

    def PositionUpdate(self, acc_data, gyro_data):
        for acc, gyro in zip(acc_data, gyro_data):
            self.OrientationUpdate(gyro)
            acc_world = self.RotateVector(acc)
            self.velocity += acc_world * self.dt
            self.position += self.velocity * self.dt

    def get_position(self):
        return self.position.tolist()

class Multipose:
    def __init__(self):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    def InitializingMedia(self):
        MFStartup = ctypes.windll.mfplat.MFStartup
        MFShutdown = ctypes.windll.mfplat.MFShutdown
        MF_VERSION = 0x00020070
        hr = MFStartup(MF_VERSION, 0)
        if hr!= 0:
            raise Exception()
        return MFShutdown
    
    def Arguments(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--device", type=int, default=0)
        self.parser.add_argument("--file", type=str, default=None)
        self.parser.add_argument("--width", type=int, default=960)
        self.parser.add_argument("--height", type=int, default=540)
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
        global args, cap_device, cap_width, cap_height, MFShutdown, mirror, keypoint_score_th, bbox_score_th, model, input_size
        args = self.Arguments()
        cap_device = args.device
        cap_width = args.width
        cap_height = args.height
        MFShutdown = self.InitializingMedia()
        if args.file is not None:
            cap_device = args.file
        mirror = args.mirror
        keypoint_score_th = args.keypoint_score
        bbox_score_th = args.bbox_score
        model_url = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
        input_size = 256
        module = tfhub.load(model_url)
        model = module.signatures['serving_default']

    def DebugDraw(self, image, elapsed_time, keypoint_score_th, keypoints_list, scores_list, bbox_score_th, bbox_list):
        self.debug_image = copy.deepcopy(image)

        for keypoints, scores in zip(keypoints_list, scores_list):
            for idx1, idx2 in [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(11,12),(5,11),(11,13),(13,15),(6,12),(12,14),(14,16)]:
                if scores[idx1] > keypoint_score_th and scores[idx2] > keypoint_score_th:
                    self.point01 = keypoints[idx1]
                    self.point02 = keypoints[idx2]
                    cv.line(self.debug_image, tuple(self.point01), tuple(self.point02), (255, 255, 255), 4)
                    cv.line(self.debug_image, tuple(self.point01), tuple(self.point02), (0, 0, 0), 2)

        for keypoints, scores in zip(keypoints_list, scores_list):
            for keypoint, score in zip(keypoints, scores):
                if score > keypoint_score_th:
                    cv.circle(self.debug_image, tuple(keypoint), 6, (255, 255, 255), -1)
                    cv.circle(self.debug_image, tuple(keypoint), 3, (0, 0, 0), -1)

        for self.bbox in bbox_list:
            if self.bbox[4] > bbox_score_th:
                cv.rectangle(self.debug_image, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (255, 255, 255), 4)
                cv.rectangle(self.debug_image, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (0, 0, 0), 2)

        cv.putText(self.debug_image, "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv.LINE_AA)
        cv.putText(self.debug_image, "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv.LINE_AA)

        return self.debug_image
        
            
            
class Animation:
    def __init__(self, **kwargs):
        self.al = kwargs['al']
        self.gl = kwargs['gl']
        self.ml = kwargs['ml']
        
        self.flat = [[item for sublist in self.al for item in sublist],
                     [item for sublist in self.gl for item in sublist],
                     [item for sublist in self.ml for item in sublist]]
        
        self.common_xlim = (0, 100)
        self.fig, self.axs = plt.subplots(3, 1)
        self.lines = {
            'axl': self.axs[0].plot([], [], 'r-')[0],
            'ayl': self.axs[0].plot([], [], 'r-')[0],
            'azl': self.axs[0].plot([], [], 'r-')[0],
            'gxl': self.axs[1].plot([], [], 'r-')[0],
            'gyl': self.axs[1].plot([], [], 'r-')[0],
            'gzl': self.axs[1].plot([], [], 'r-')[0],
            'mxl': self.axs[2].plot([], [], 'r-')[0],
            'myl': self.axs[2].plot([], [], 'r-')[0],
            'mzl': self.axs[2].plot([], [], 'r-')[0],
            
        }
        
        for ax in self.axs:
            ax.set_xlim(self.common_xlim)
        
        self.tlist = []
    
    def update(self, frame):
        self.axs[0].set_ylim((min(self.flat[0]), max(self.flat[0])))
        self.axs[1].set_ylim((min(self.flat[1]), max(self.flat[1])))
        self.axs[2].set_ylim((min(self.flat[2]), max(self.flat[2])))
        
        self.axl = [k[0] for k in self.al]
        self.ayl = [k[1] for k in self.al]
        self.azl = [k[2] for k in self.al]
        self.gxl = [k[0] for k in self.ml]
        self.gyl = [k[1] for k in self.ml]
        self.gzl = [k[2] for k in self.ml]
        self.mxl = [k[0] for k in self.gl]
        self.myl = [k[1] for k in self.gl]
        self.mzl = [k[2] for k in self.gl]
        self.tlist.append(frame)
        
        if len(self.tlist) > 1:
            self.new_xlim = (self.tlist[-1]-self.tlist[0], self.tlist[-1])
            for ax in self.axs:
                ax.set_xlim(self.new_xlim)
        self.lines['axl'].set_data(self.tlist, self.axl)
        self.lines['ayl'].set_data(self.tlist, self.ayl)
        self.lines['azl'].set_data(self.tlist, self.azl)
        self.lines['gxl'].set_data(self.tlist, self.gxl)
        self.lines['gyl'].set_data(self.tlist, self.gyl)
        self.lines['gzl'].set_data(self.tlist, self.gzl)
        self.lines['mxl'].set_data(self.tlist, self.mxl)
        self.lines['myl'].set_data(self.tlist, self.myl)
        self.lines['mzl'].set_data(self.tlist, self.mzl)

        return self.lines.values()
    
    def main(self):
        self.ani = animation.FuncAnimation(self.fig, self.__class__.update, frames=range(1000), blit=True, interval = 100)
        plt.tight_layout()
        plt.show()
        
        print(self)

def KeypointBound(keypoint: int) -> bool:
    x, y = keypoint
    return x<0

def FloorChecker(height: float) -> bool:
    global Positioned
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
    global Positioned
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


async def scan():
    global devicelist
    devicelist = []
    
    devices = await BleakScanner.discover()
    for device in devices:
        print(device)
        devicelist.append(device.address[:18])
        
def sqrt(x: float) -> float:
    return x**0.5

def mod(x: float, y: float) -> float:
    return x%y

def average(x: list) -> float:
    return sum(x)/len(x)

def S2D(scientific_str: any) -> any:
    decimal_value = float(scientific_str)
    return format(decimal_value, 'f')

def RMS(l: list) -> float:
    return sqrt(sum(average([x**2 for x in l])))


asyncio.run(scan())

print('a')

# Port=int(input())
#def inp():
#    global Port, ser
#    def inp2():
#        global Port
#        try:
#            print("Port : ",end = ' ')
#            Port=int(input())
#        except:
#            print('Invalid Port Number')
#            inp2()
#    inp2()
#
#    def defserial():
#        global ser
#        try:
#            ser = serial.Serial("COM{}".format(Port), 2000000, timeout=10)
#            print(ser)
#        except Exception as e:
#            print(e)
#            print('type \'continue\' to continue...')
#            if input() == 'continue':
#                inp()
#            else:
#                print('retry...')
#                defserial()
#    defserial()  
#
#inp()


# a = float(input())
# x, y, floor = map(float, input().split())
async def get_device():
    global nano_device
    devices = await BleakScanner.discover()
    nano_device = None
    for device in devices:
        for service in device.metadata['uuids']:
            if SERVICE_UUID in service:
                nano_device = device
                break
    print(nano_device)
asyncio.run(get_device())

al = []
gl = []
ml = []

async def getDefault():
    async with BleakClient(nano_device.address) as client:
        print("Connected: {client.is_connected}")
        def Callback1(sender, data):
            global al
            ax, ay, az = data[:4], data[4:8], data[8:]
            al.append([ax, ay, az])

        def Callback2(sender, data):
            global gl
            gx, gy, gz = data[:4], data[4:8], data[8:]
            gl.append([gx, gy, gz])

        def Callback3(sender, data):
            global ml
            mx, my, mz = data[:4], data[4:8], data[8:]
            ml.append([mx, my, mz])

        await client.start_notify(ACCEL_CHARACTERISTIC_UUID, Callback1)
        await client.start_notify(GYRO_CHARACTERISTIC_UUID, Callback2)
        await client.start_notify(MAG_CHARACTERISTIC_UUID, Callback3)

        await client.stop_notify(ACCEL_CHARACTERISTIC_UUID)
        await client.stop_notify(GYRO_CHARACTERISTIC_UUID)
        await client.stop_notify(MAG_CHARACTERISTIC_UUID)
asyncio.run(getDefault())

default_unpacked = al[0] + gl[0] + ml[0]

# default = ser.read(36)
# default_unpacked = struct.unpack('<9f',default)
# default_unpacked = list(default_unpacked)
default_unpacked[0] /= 16384
default_unpacked[1] /= 16384
default_unpacked[2] /= 16384
default_unpacked[3] /= 131
default_unpacked[4] /= 131
default_unpacked[5] /= 131

# a = mod(((default_unpacked[3] + default_unpacked[4])/2), 360)
a = mod(average([default_unpacked[3], default_unpacked[4]]))
x, y, floor = default_unpacked[-3], default_unpacked[-2], default_unpacked[-1]
print(a, x, y, floor, sep='\n')

i = 0


dx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
dy = [7, 4, -4, -7]
PointList = []
while (len(PointList)!=16):
    PointList.append([(a/180)*dx[i] if a!=0 else dx[i], (a/180)*dy[i//4] if a!=0 else dy[i//4]])
    i+=1

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out"]
Positioned = [False]*len(Positioned)
responseList = []
t = 0
tlist= [0]

pose = Multipose()
pose.main()
calcPose = Position(ml[-1],[0,0,0] ,0.001)
animate = Animation(al = al, gl = gl, ml = ml)

cap = cv.VideoCapture(0)
cap.open(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
temp = 0
def devicenotfound():    
    if not nano_device:
        print("Device not found")
        time.sleep(5000)
        devicenotfound()
        
async def main():
    while True:
        st = time.time()
        
        ret, frame = cap.read()
        if not ret:
            if temp > 3:
                break
            print(f't{temp}')
            time.sleep(1)
            temp += 1
            continue
        if mirror:
            frame = cv.flip(frame, 1)  
        debug_image = copy.deepcopy(frame)
        keypoints_list, scores_list, bbox_list = pose.Inference(model, input_size, frame)

        key = cv.waitKey(1)
        if key == 27:  
            break
        cv.imshow('cam1', debug_image)

        # response = ser.read(36)
        # unpacked = struct.unpack('<9f', response)
        # unpacked = list(unpacked)
        # unpacked[0] /= 16384
        # unpacked[1] /= 16384
        # unpacked[2] /= 16384
        # unpacked[3] /= 131
        # unpacked[4] /= 131
        # unpacked[5] /= 131
        # unpacked = tuple(unpacked)
        # responseList.append(unpacked)
        # accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ = unpacked[0],unpacked[1],unpacked[2],unpacked[3],unpacked[4],unpacked[5],unpacked[6],unpacked[7],unpacked[8]

        # calcPose.PositionUpdate(al, gl)

        # al.append([accelX, accelY, accelZ])
        # gl.append([gyroX, gyroY, gyroZ])
        # ml.append(calcPose.get_position())
        # print(responseList[-1])

        async with BleakClient(nano_device.address) as client:
            print("Connected: {client.is_connected}")
            def Callback1(sender, data):
                global al
                ax, ay, az = data[:4], data[4:8], data[8:]
                al.append([ax, ay, az])
                
            def Callback2(sender, data):
                global gl
                gx, gy, gz = data[:4], data[4:8], data[8:]
                gl.append([gx, gy, gz])
            
            def Callback3(sender, data):
                global ml
                mx, my, mz = data[:4], data[4:8], data[8:]
                ml.append([mx, my, mz])
                
            await client.start_notify(ACCEL_CHARACTERISTIC_UUID, Callback1)
            await client.start_notify(GYRO_CHARACTERISTIC_UUID, Callback2)
            await client.start_notify(MAG_CHARACTERISTIC_UUID, Callback3)
            
            await client.stop_notify(ACCEL_CHARACTERISTIC_UUID)
            await client.stop_notify(GYRO_CHARACTERISTIC_UUID)
            await client.stop_notify(MAG_CHARACTERISTIC_UUID)
            
        accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ = al[-1][0], al[-1][1], al[-1][2], gl[-1][0], gl[-1][1], gl[-1][2], ml[-1][0], ml[-1][1], ml[-1][2]
        responseList.append([accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ])
        try:
            if OutLinedChecker(magX,magY):
                print('ball outlined')
            if FloorChecker(magZ) and HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
                continue
            if HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ):
                state = RootCheckerRecursion([accelX,accelY,accelZ], 0, 0, True)
                if state == True:
                    print('r player hitted by l player')
                elif state == False:
                    print('l player hitted by r player')
                else:
                    pass
        except Exception as exception:
            print(exception)
            
        elapsed = await time.time()-st
        tlist.append(tlist[-1]+elapsed)
        t+=elapsed
        debug_image = pose.DebugDraw(
            debug_image,
            elapsed,
            keypoint_score_th,
            keypoints_list,
            scores_list,
            bbox_score_th,
            bbox_list
        )
asyncio.run(main())
