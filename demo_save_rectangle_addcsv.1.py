#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import time
import csv
import os
from timeit import time
import warnings
import sys
import numpy.core.multiarray
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import keyboard
warnings.filterwarnings('ignore')

trackId_list=[]

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output_test_cross3.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        #setup the localtime
        localtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            tracker_Id=int(track.track_id)
            if tracker_Id not in trackId_list:
                trackId_list.append(int(track.track_id))
                globals()['trackerId'+str(tracker_Id)]=[]
            print("trackId_list=",trackId_list)
            if tracker_Id in trackId_list and keyboard.is_pressed('g'):# if key 'g' is pressed 
                print('You Pressed G Key to save coordinate!') #some bug
                w_2=int(bbox[0])+(int(bbox[2])-int(bbox[0]))/2
                globals()['trackerId'+str(tracker_Id)].append([w_2,int(bbox[1])])
                print('trackerId'+str(tracker_Id),globals()['trackerId'+str(tracker_Id)])
                
                # Save file to excel with csv in output
                with open('./output_process_0416.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([localtime, 'trackerId'+str(tracker_Id),globals()['trackerId'+str(tracker_Id)]])
                # Save XY to excel with csv in outputXY
                with open('./output_XY_0416.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([w_2, int(bbox[1])])

        
        for i in trackId_list:
            print('trackerId'+str(i),globals()['trackerId'+str(i)])

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save file to excel with csv in output
            with open('./output_test_cross4.csv', 'a') as csvfile:
                for i in trackId_list:
                    writer = csv.writer(csvfile)
                    writer.writerow(['trackerId'+str(i), globals()['trackerId'+str(i)]])
                    # writer.writerow([globals()['trackerId'+str(i)]]) #id
                    # writer.writerow(['trackerId'+str(i)]) #xy
                    #csvfile.write('\n')
            break
        
        

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
