# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals
import copy
import time
import argparse
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import kagglehub

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true', default=True)

    parser.add_argument("--keypoint_score", type=float, default=0.4)
    parser.add_argument("--bbox_score", type=float, default=0.3)

    args = parser.parse_args()

    return args


def run_inference(model, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    
    input_image = cv.resize(image, dsize=(input_size, input_size))  
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  
    input_image = input_image.reshape(-1, input_size, input_size, 3)  
    input_image = tf.cast(input_image, dtype=tf.int32)  

    
    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    
    keypoints_list, scores_list = [], []
    bbox_list = []
    for keypoints_with_score in keypoints_with_scores:
        keypoints = []
        scores = []
        
        for index in range(17):
            keypoint_x = int(image_width *
                             keypoints_with_score[(index * 3) + 1])
            keypoint_y = int(image_height *
                             keypoints_with_score[(index * 3) + 0])
            score = keypoints_with_score[(index * 3) + 2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)

        
        bbox_ymin = int(image_height * keypoints_with_score[51])
        bbox_xmin = int(image_width * keypoints_with_score[52])
        bbox_ymax = int(image_height * keypoints_with_score[53])
        bbox_xmax = int(image_width * keypoints_with_score[54])
        bbox_score = keypoints_with_score[55]

        
        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append(
            [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def main():
    
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    keypoint_score_th = args.keypoint_score
    bbox_score_th = args.bbox_score

    cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
    cap.open(cap_device)
    ##cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    model_url = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
    input_size = 256

    module = tfhub.load(model_url)
    model = module.signatures['serving_default']
    temp = 0
    while True:
        start_time = time.time()

        
        ret, frame = cap.read()
        if not ret:
            if temp>3:
                break
            print('a')
            time.sleep(1)
            temp+=1
            continue
        if mirror:
            frame = cv.flip(frame, 1)  
        debug_image = copy.deepcopy(frame)

        
        keypoints_list, scores_list, bbox_list = run_inference(
            model,
            input_size,
            frame
        )

        elapsed_time = time.time() - start_time

        
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints_list,
            scores_list,
            bbox_score_th,
            bbox_list
        )

        
        key = cv.waitKey(1)
        if key == 27:  
            break

        cv.imshow('cam1', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints_list,
    scores_list,
    bbox_score_th,
    bbox_list
):
    debug_image = copy.deepcopy(image)

    
    
    for keypoints, scores in zip(keypoints_list, scores_list):
        
        index01, index02 = 0, 1
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 0, 2
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 1, 3
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 2, 4
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 0, 5
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 0, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 5, 6
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 5, 7
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 7, 9
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 6, 8
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 8, 10
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 11, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 5, 11
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 11, 13
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 13, 15
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 6, 12
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 12, 14
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)
        
        index01, index02 = 14, 16
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, (255, 255, 255), 4)
            cv.line(debug_image, point01, point02, (0, 0, 0), 2)

        
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
                cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    
    for bbox in bbox_list:
        if bbox[4] > bbox_score_th:
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 255, 255), 4)
            cv.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 0), 2)

    
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image

main()
