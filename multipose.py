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
import ctypes
import sympy

cv.destroyAllWindows()


class Multipose:
    
    def __init__(self):
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    def initialize_media_foundation(self):
        MFStartup = ctypes.windll.mfplat.MFStartup
        MFShutdown = ctypes.windll.mfplat.MFShutdown
        MF_VERSION = 0x00020070

        hr = MFStartup(MF_VERSION, 0)
        if hr != 0:
            raise Exception('hr!=0')
        return MFShutdown
    
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
        self.args = self.get_args()
        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height
        self.MFShutdown = self.initialize_media_foundation()
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
            keypoints_list, scores_list, bbox_list = self.run_inference(self.model, self.input_size, self.frame)
            self.elapsed_time = time.time() - self.start_time
            self.debug_image = self.draw_debug(
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

    def draw_debug(self, image, elapsed_time, keypoint_score_th, keypoints_list, scores_list, bbox_score_th, bbox_list):
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
    
    
def KeypointBound(keypoint: int):
    x, y = keypoint
    return x<0
mp = Multipose()
mp.main()
