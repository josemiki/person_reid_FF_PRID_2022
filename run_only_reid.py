import numpy as np
import cv2
#import cuhk03_dataset
import time
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import copy
import sys

from reid.person_reid import personReIdentifier

def getFrameNumber(count):
    count = count + 1
    return count

def pReID_queries_per_videos(reidentifier, queries_path, video_path, topN, frame_seq ):
    
    fpath, fname = os.path.split(video_path)
    seq_path = fpath+'/seq_videos'
    seq_videos = sorted(glob.glob(seq_path+'/*'))
    print("len(seq_videos): ",len(seq_videos))
    for v in seq_videos:
        reidentifier.PersonReIdentification(v, topN, v, queries_path)

if __name__ == '__main__':
        print('dataset y ReID')
        
        reidentifier = personReIdentifier()

        data_dir='/home/josemiki/FF-PRID-2020/data'

        testAB = sorted(glob.glob(data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(data_dir + '/B-A/*'))

        topN = 100
        t_frames_seq = 100
        
        print("############# Run2.py configuration ####################")
        print("data_dir:",data_dir)
        print("len(testAB): ",len(testAB))
        print("len(testBA): ",len(testBA))
        print("topN: ",topN)
        print("t_frames_seq: ",t_frames_seq)
        print("########################################################")

        # inside every sub_carpet in testAB, there are:
        # ground_truth.csv
        # video_in.avi 
        # queries(4): person_/*.png

        print("Begin CROPPER AND REID A -> B **************************************************")
        for carpet_test_path in testAB:
            pReID_queries_per_videos(reidentifier, carpet_test_path , carpet_test_path + '/video_in.avi', topN, t_frames_seq)
        print("END CROPPER AND REID A -> B **************************************************")
        
        print("################################################################################")
        
        print("BEGIN CROPPER AND REID B -> A **************************************************")
        for carpet_test_path in testBA:
            pReID_queries_per_videos(reidentifier, carpet_test_path , carpet_test_path + '/video_in.avi', topN,t_frames_seq)
        print("END CROPPER AND REID B -> A **************************************************")
        