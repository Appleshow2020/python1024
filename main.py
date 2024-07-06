# -*- coding : utf-8 -*-
# pylint: disable=invalid_name,too-many-instance-attributes, too-many-arguments

from __future__ import (absolute_import, division, unicode_literals)
import sys, os
import serial as s
import math
import copy
import time
import argparse
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import kagglehub


class Multipose:
    def __init__(self):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = 0

    
    def get_args(self):
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


    def run_inference(self, model, input_size, image):
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
                self.keypoint_x = int(self.image_width *
                                 i[(index * 3) + 1])
                self.keypoint_y = int(self.image_height *
                                 i[(index * 3) + 0])
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

        self.args = self.get_args()
        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height

        if self.args.file is not None:
            self.cap_device = self.args.file

        self.mirror = self.args.mirror
        self.keypoint_score_th = self.args.keypoint_score
        self.bbox_score_th = self.args.bbox_score
        self.cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
        self.cap.open(self.cap_device)
        ##self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        self.model_url = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
        self.input_size = 256

        self.module = tfhub.load(self.model_url)
        self.model = self.module.signatures['serving_default']
        self.temp = 0
        while True:
            self.start_time = time.time()
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                if self.temp>3:
                    break
                print('a')
                time.sleep(1)
                self.temp+=1
                continue
            if self.mirror:
                self.frame = cv.flip(self.frame, 1)  
            self.debug_image = copy.deepcopy(self.frame)


            keypoints_list, scores_list, bbox_list = self.run_inference(
                self.model,
                self.input_size,
                self.frame
            )

            self.elapsed_time = time.time() - self.start_time


            self.debug_image = self.draw_debug(
                self.debug_image,
                self.elapsed_time,
                self.keypoint_score_th,
                self.keypoints_list,
                self.scores_list,
                self.bbox_score_th,
                self.bbox_list
            )


            self.key = cv.waitKey(1)
            if self.key == 27:  
                break

            cv.imshow('cam1', self.debug_image)

        self.cap.release()
        cv.destroyAllWindows()


    def draw_debug(self, image, elapsed_time, keypoint_score_th, keypoints_list, scores_list, bbox_score_th, bbox_list):  

        self.debug_image = copy.deepcopy(image)

        for keypoints, scores in zip(keypoints_list, scores_list):

            index01, index02 = 0, 1
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 0, 2
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 1, 3
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 2, 4
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 0, 5
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 0, 6
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 5, 6
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 5, 7
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 7, 9
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 6, 8
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 8, 10
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 11, 12
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 5, 11
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 11, 13
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 13, 15
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 6, 12
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 12, 14
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)

            index01, index02 = 14, 16
            if scores[index01] > keypoint_score_th and scores[
                    index02] > keypoint_score_th:
                self.point01 = keypoints[index01]
                self.point02 = keypoints[index02]
                cv.line(self.debug_image, self.point01, self.point02, (255, 255, 255), 4)
                cv.line(self.debug_image, self.point01, self.point02, (0, 0, 0), 2)


            for keypoint, score in zip(keypoints, scores):
                if score > keypoint_score_th:
                    cv.circle(self.debug_image, keypoint, 6, (255, 255, 255), -1)
                    cv.circle(self.debug_image, keypoint, 3, (0, 0, 0), -1)


        for bbox in self.bbox_list:
            if bbox[4] > bbox_score_th:
                cv.rectangle(self.debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             (255, 255, 255), 4)
                cv.rectangle(self.debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                             (0, 0, 0), 2)


        cv.putText(self.debug_image,
                   "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
                   cv.LINE_AA)
        cv.putText(self.debug_image,
                   "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                   cv.LINE_AA)

        return self.debug_image


def FloorChecker(height):
    if height == floor:
        Positioned[0] = True
        return True
    else:
        Positioned[0] = False
        return False
def HittedChecker(ax1, ay1, az1, ax2, ay2, az2, mz): 
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

def ThrowerChecker(l,temp): 
    try:
        a = math.sqrt((l[0]**2)+(l[1]**2)+(l[2]**2))
        al.append([a,l[3],l[4],l[5]])
        return ThrowerCheckerRecursion(temp,0)
    except ValueError:
        return None

def ThrowerCheckerRecursion(i,p): 
    try:
        if p<=8 and p>=0:
            if all(al[i - j][0] < al[i - j + 1][0] for j in range(10 - p, 1, -1)):
                return ThrowerCheckerRecursion(i - (9 - p), p)
            else:
                return ThrowerCheckerRecursion(i, p + 1)
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
        Positioned[3] = True
        return True
    Positioned[3] = False
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
ser.writebytes(f'Connected as Port: {ser.portstr}, baudrate:{ser.baudrate}', encoding='ascii')
Positioned = ["On Floor", "Hitted", "Thrower", "OutLined", "L In", "R In", "L Out", "R Out"]
Positioned = [False]*len(Positioned)
responseList = []
al = []
while True:
    response = ser.readline().decode()
    responseList.append(list(response.split(',')))
    accelX, accelY, accelZ, gyroX, gyroY, gyroZ, magX, magY, magZ = response.split(',')
    FloorChecker(magZ)
    HittedChecker(accelX,accelY,accelZ,responseList[-2][0],responseList[-2][1],responseList[-2][2],magZ)
    ThrowerChecker(accelX,accelY,accelZ,0)
    OutLinedChecker(magX,magY)

    Pose = Multipose()
    
