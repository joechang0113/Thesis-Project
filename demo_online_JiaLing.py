#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import time
import csv
import os
# from timeit import time
import warnings
import sys
import numpy.core.multiarray
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
# 2019-05-10 14: 52.xls
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import keyboard
import datetime
# import time

warnings.filterwarnings('ignore')

trackId_list = []


def main(yolo):

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine",
                                                       max_cosine_distance,
                                                       nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./Output_info_Joe/online_2.avi', fourcc, 15,
                              (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    number = 0
    while True:
        number = number + 1
        # print("number:", number)
        ret, frame = video_capture.read()  # frame shape 640*480*3 #ret..r
        if ret is not True:
            break
        t1 = time.time()  # t1..start_time

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [
            Detection(bbox, 1.0, feature)
            for bbox, feature in zip(boxs, features)
        ]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap,
                                                    scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # pre_w_2 = w_2
        # print("localtime:", localtime)
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id),
                        (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                        (0, 255, 0), 2)
            tracker_Id = int(track.track_id)
            if tracker_Id not in trackId_list:
                trackId_list.append(int(track.track_id))
                globals()['trackerId' + str(tracker_Id)] = []
                globals()['trackerId' + str(tracker_Id) + '_time'] = []
                globals()['timeId' + str(tracker_Id)] = []
            print("trackId_list=", trackId_list)

            cur_w_2 = int(bbox[0]) + abs((int(bbox[2]) - int(bbox[0])) / 2)
            if number % 10 == 0:
                localtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime())
                # localtime = time.strftime("%H:%M:%S", time.localtime())
                print("localtime:", localtime)
                print('save coordinate now !!!')  # some bug
                w_2 = int(bbox[0]) + abs((int(bbox[2]) - int(bbox[0])) / 2)
                # pre_w_2 = w_2
                movecheck = abs(int(cur_w_2) - int(w_2))

                # ## move check ###

                if movecheck > 3:
                    globals()['trackerId' + str(tracker_Id)].append(["1"])
                else:
                    globals()['trackerId' + str(tracker_Id)].append(["0"])

                print(movecheck)
                print(cur_w_2)
                print(w_2)
                w_2 = cur_w_2
                # ## move check ###

                globals()['trackerId' + str(tracker_Id)].append(
                    [w_2, int(bbox[1])])
                globals()['trackerId' + str(tracker_Id) +
                          '_time'].append(localtime)
                globals()['timeId' + str(tracker_Id)].append(localtime)
                # globals()['trackerId'+str(tracker_Id)].append([int(bbox[0]),int(bbox[2])])
                print('trackerId' + str(tracker_Id),
                      globals()['trackerId' + str(tracker_Id)])
                print('trackerId' + str(tracker_Id) + '_time',
                      globals()['trackerId' + str(tracker_Id) + '_time'])

                # Save file to excel with csv in output
                # with open('./output_process_0510_4.csv', 'a') as csvfile:
                # writer = csv.writer(csvfile)
                # writer.writerow([localtime, 'trackerId'+str(tracker_Id),globals()['trackerId'+str(tracker_Id)]])
                # Save XY to excel with csv in outputXY
                # with open('./output_XYBegin_0522.csv', 'w') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(['XYBegin', globals()['trackerId'+str(tracker_Id)]])

        for i in trackId_list:
            print('trackerId' + str(i), globals()['trackerId' + str(i)])
            # print('trackerId'+str(i), globals()['trackerId'+str(i)])
            #####
            # if globals()[str(i)-str(i-1)]>5:
            #     print("Moving.....(1)")
            # else:
            #     print("Stop.....(0)")
            #####

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' +
                        str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save file to excel with csv in output
            with open('./Output_info_Joe/online_2.csv', 'w') as csvfile:
                for i in trackId_list:
                    # globals()['time_'].append(localtime)
                    writer = csv.writer(csvfile)
                    # writer.writerow([('trackerId'+str(i)), "Time")
                    writer.writerow([
                        'trackerId' + str(i),
                        globals()['trackerId' + str(i)]
                    ])
                    writer.writerow(
                        ['timeId' + str(i),
                         globals()['timeId' + str(i)]])
                    # writer.writerow([globals()['trackerId'+str(i)]]) #id
                    # writer.writerow(['trackerId'+str(i)]) #xy
                    # csvfile.write('\n')
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
