# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments
from __future__ import (absolute_import, division, unicode_literals)
# import serial
import time
import struct
# import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import asyncio
from bleak import BleakScanner, BleakClient
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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
    
class Animation:
    def __init__(self, al,gl,ml):
        self.al=al;self.gl=gl;self.ml=ml
        
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

al = {}
gl = {}
ml = {}

async def get_device():
    global nano_device
    nano_device = None
    temp=0
    flag=False
    while nano_device==None:
        devices = await BleakScanner.discover()
        for device in devices:
            for service in device.metadata['uuids']:
                if SERVICE_UUID in service:
                    nano_device = device
                    flag=True
                    break
            if flag:
                break
        temp+=1
        print(temp)
    print(nano_device)

async def getDefault():
    async with BleakClient(nano_device.address) as client:
        print("Connected: {client.is_connected}")
        def Callback1(_, data):
            global al
            ax, ay, az = data[:4], data[4:8], data[8:]
            al[time.perf_counter_ns()] = {"x":ax,"y":ay,"z":az}
            print("Collecting datas.")

        def Callback2(_, data):
            global gl
            gx, gy, gz = data[:4], data[4:8], data[8:]
            gl[time.perf_counter_ns()] = {"x":gx,"y":gy,"z":gz}
            print("Collecting datas..")

        def Callback3(_, data):
            global ml
            mx, my, mz = data[:4], data[4:8], data[8:]
            ml[time.perf_counter_ns()] = {"x":mx,"y":my,"z":mz}
            print("Collecting datas...")

        await client.start_notify(ACCEL_CHARACTERISTIC_UUID, Callback1)
        await client.start_notify(GYRO_CHARACTERISTIC_UUID, Callback2)
        await client.start_notify(MAG_CHARACTERISTIC_UUID, Callback3)
        await asyncio.sleep(3)
        await client.stop_notify(ACCEL_CHARACTERISTIC_UUID)
        await client.stop_notify(GYRO_CHARACTERISTIC_UUID)
        await client.stop_notify(MAG_CHARACTERISTIC_UUID)


start = input("press any key to start collect data")
asyncio.run(get_device())
asyncio.run(getDefault())
del start
default_unpacked = list(al[max(al.keys())].values())+list(gl[max(gl.keys())].values())+list(ml[max(ml.keys())].values())

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
a = mod(average([default_unpacked[3], default_unpacked[4]]),360)
x, y, floor = default_unpacked[-3], default_unpacked[-2], default_unpacked[-1]
print(a, x, y, floor, sep='\n')

i = 0

dx = [-11,-4,4,11,-8,-4,4,8,-8,-4,4,8,-11,-4,4,11]
dy = [7, 4, -4, -7]
PointList = []
while (len(PointList)!=16):
    PointList.append([(a/180)*dx[i] if a!=0 else dx[i], (a/180)*dy[i//4] if a!=0 else dy[i//4]])
    i+=1
print("Finished collecting default datas.")

Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out"]
Positioned = [False]*len(Positioned)
responseList = []
t = 0
tlist= [0]

calcPose = Position(ml[-1],[0,0,0] ,0.001)
animate = Animation(al = al, gl = gl, ml = ml)

temp = 0
def devicenotfound():    
    if not nano_device:
        print("Device not found")
        exit(0)
        
async def main():
    data_queue = Queue()
    executor = ThreadPoolExecutor()

    def notification_handler(sender, data):
        timestamp = time.perf_counter_ns()
        data_queue.put((timestamp, sender, data))

    def process_data():
        while True:
            if not data_queue.empty():
                timestamp, sender, data = data_queue.get()
                print(f"Processed {timestamp}: {sender} → {data.hex()}")

    async def ble_task(address, char_uuid):
        from bleak import BleakClient
        async with BleakClient(address) as client:
            await client.start_notify(char_uuid, notification_handler)
            print("BLE Notification Listening Started.")

            loop = asyncio.get_running_loop()
            loop.run_in_executor(executor, process_data)

            try:
                while True:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass
            finally:
                await client.stop_notify(char_uuid)
                print("BLE Notification Stopped.")

    ble = asyncio.create_task(ble_task(nano_device, SERVICE_UUID))
    
    while True:
        st = time.time()
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
    await ble
    executor.shutdown()
asyncio.run(main())
