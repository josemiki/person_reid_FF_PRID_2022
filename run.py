import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import copy
import sys
#------------------------------->
#same yolo_cropper as first paper
from pYOLO import yolo_cropper
#------------------------------->
#First Re-ID model on first paper
#from pReID import personReID
#New Re-ID model using fast_reid
from reid.person_reid import personReIdentifier
#------------------------------->
from matplotlib.widgets import TextBox
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'demo', 'Mode graph, val, demo, test')
#
tf.flags.DEFINE_string('query_path', '', 'First image path to compare')
tf.flags.DEFINE_string('cropps_path', '', 'gallery')
#
tf.flags.DEFINE_string('queries_path','','carpet of queries path to compare')
tf.flags.DEFINE_string('video_path','','video path to cropping')
tf.flags.DEFINE_string('data_dir', '../data/dataReal_pt1', 'path to dataset')#DATASET
tf.flags.DEFINE_string('p_name', 'predictV2', 'name path of predict file')
#
tf.flags.DEFINE_string('t_skip','' ,'t is number of frames per sequence')
tf.flags.DEFINE_string('beta','' ,'beta is the threshold')
tf.flags.DEFINE_string('eta','' ,'eta is the TOP')
#
tf.flags.DEFINE_string('threshold', '0.0' ,'threshold for True reid')
tf.flags.DEFINE_integer('top', '20' ,'Top for reid')
#
tf.flags.DEFINE_string('graph', '' ,'Top for reid')
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160

def getFrameNumber(count):
    count = count + 1
    return count

def pReID_queries_per_videos(cropper, reidentifier, queries_path, video_path, topN, frame_seq ):
    queries = sorted(glob.glob(queries_path+'/*.png'))
    print(len(queries))
    ##-------------------------------> MAKE SEQUENCES
    fpath, fname = os.path.split(video_path )
    seq_path = fpath+'/seq_videos'
    if not os.path.isfile(seq_path):
        os.system('mkdir '+ seq_path)
    carpet = 0
    cont_seq=0
    #n_carpet=''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while(True):
        ret, frame = cap.read()
        if ret == True:
            height, width, layers = frame.shape
            size = (width,height)
            if(cont_seq <= frame_seq ):
                if(cont_seq==frame_seq):
                    cont_seq = 0
                    carpet = carpet+1
                
                if(cont_seq==0):
                    ncarpet = '{0:06}'.format(getFrameNumber(carpet))                    
                    os.system('mkdir '+ seq_path+'/'+ncarpet)
                    pathOut = seq_path+'/'+ncarpet+'/sub_video.avi'
                    out = cv2.VideoWriter(pathOut , cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                out.write(frame)
                cont_seq = cont_seq + 1
        else:
            break
    ##-------------------------------> END MAKE SEQUENCES
    seq_videos = sorted(glob.glob(seq_path+'/*'))
    for v in seq_videos:
        fpath_cropps = v + '/cropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        cropper.personCropping( v+'/sub_video.avi', fpath_cropps)
        reidentifier.PersonReIdentification(v, topN, v, queries_path)

key_rank = ''

def submit_rank(text):
    global key_rank
    
    #if(text != '' and int(text)):
    if(text == 'auto'):
        key_rank = 'auto'
        plt.close()
    elif(int(text) == 0):
        #print('QUERY ISN\'T IN RANK')
        key_rank = text
        plt.close()
    elif(int(text) > 0 and int(text) <= int(FLAGS.top)):
        key_rank = text
        plt.close()
    else:
        print('error: isn\'t number')

def getMetrics_TVR_FR(dict_valuesAB, dict_valuesBA, l_predictAB, l_predictBA):
    tc, tmc, fs, fc, ts = 0,0,0,0,0
    FR_l, TVR_l = [], []
    
    dict_predictAB = copy.deepcopy(dict_valuesAB)
    dict_predictBA = copy.deepcopy(dict_valuesBA)

    for elem in l_predictAB:
        tup = tuple(elem)
        idpath, idname = os.path.split(tup[1])
        seqpath, seqname = os.path.split(idpath)
        l_id = idname.split('_')
        id_t = int( l_id[len(l_id)-1])
        seq_t = int(seqname)
        high , gt, rank, t_score = tup[2], tup[3], tup[4], tup[5] 
        try:
            dict_predictAB[id_t][seq_t] = (high, gt, rank, t_score) # en el id, con seq
        except:
            continue

    for elem in l_predictBA:
        tup = tuple(elem)
        idpath, idname = os.path.split(tup[1])
        seqpath, seqname = os.path.split(idpath)
        l_id = idname.split('_')
        id_t = int( l_id[len(l_id)-1])
        seq_t = int(seqname)
        high , gt, rank, t_score = tup[2], tup[3], tup[4], tup[5] 
        try:
            dict_predictBA[id_t][seq_t] = (high, gt, rank, t_score) # en el id, con seq
        except:
            continue
    
    ###### CALCULANDO ... TC, TMC, FS, FC, TS 
    for key_query_ID in dict_valuesAB:        
        #print('key: ', key_query_ID)
        for key_seq in dict_valuesAB[key_query_ID]:
            #tup_t = (bool(> thresh_score), GT, rank, true score)
            FR, TVR = 0, 0
            tup_t = dict_predictAB[int(key_query_ID)][key_seq]
            if(tup_t[1] == True):           # TRUES , if GT is True
                if(tup_t[0] == True and tup_t[2] > 0): # TC , score > threshold and (is in TOP10) 
                    tc = tc+1
                elif(tup_t[0] == True and tup_t[2] == 0): # TMC , score > threshold  and (is'n in TOP10)
                    tmc = tmc + 1
                elif(tup_t[0] == True and tup_t == -1):# Cuando no arroja NADA, cuento TMC
                    tmc = tmc + 1
                elif(tup_t[0] == False): #FS, score < threshold
                    fs = fs + 1
                    #AUMENTAR **** FR *****
                    #if(tc!= 0):
                    #    FR = (tc) / (tc + tmc + fs)
                    #    TVR = (tc) /(tc + tmc + fc)
                    #    if(FR != 0 and TVR != 0 and FR != 1 and TVR != 1):
                    #        FR_l.append(FR)
                    #        TVR_l.append(TVR)


                #elif(tup_t == -1):#cuando no arroja NADA
                    #continue

            elif(tup_t[1] == False) :    # FALSES , if GT is False
                if(tup_t[0] == True): # FC (score > threshol
                    fc = fc+1
                    #AUMENTAR **** TVR  ******
                    #if(tc!= 0):
                    #    FR = (tc) / (tc + tmc + fs)
                    #    TVR = (tc) /(tc + tmc + fc)
                    #    if(FR != 0 and TVR != 0 and FR != 1 and TVR != 1):
                    #        FR_l.append(FR)
                    #        TVR_l.append(TVR)
                elif(tup_t[0] == True and tup_t == -1):#Cuando no arroja NADA cuento FC
                    fc = fc+1
                elif(tup_t[0] == False): # TS score < threshold
                    ts = ts + 1
                #elif(tup_t == -1):
                    #continue       # VOID, reid no arrojo NADA
            
            #            
    ###### CALCULANDO ... TC, TMC, FS, FC, TS 
    
    for key_query_ID in dict_valuesBA:        
        #print('key: ', key_query_ID)
        for key_seq in dict_valuesBA[key_query_ID]:
            #tup_t = (bool(> thresh_score), GT, rank, true score)
            FR, TVR = 0, 0
            tup_t = dict_predictBA[int(key_query_ID)][key_seq]
            if(tup_t[1] == True):           # TRUES , if GT is True
                if(tup_t[0] == True and tup_t[2] > 0): # TC , score > threshold and (is in TOP10) 
                    tc = tc+1
                elif(tup_t[0] == True and tup_t[2] == 0): # TMC , score > threshold  and (is'n in TOP10)
                    tmc = tmc + 1
                elif(tup_t[0] == True and tup_t == -1):# Cuando no arroja NADA, cuento TMC
                    tmc = tmc + 1
                elif(tup_t[0] == False): #FS, score < threshold
                    fs = fs + 1
                
            elif(tup_t[1] == False) :    # FALSES , if GT is False
                if(tup_t[0] == True): # FC (score > threshol
                    fc = fc+1
                elif(tup_t[0] == True and tup_t == -1):#Cuando no arroja NADA cuento FC
                    fc = fc+1
                elif(tup_t[0] == False): # TS score < threshold
                    ts = ts + 1                
    
    #################################################################
    #print('metrics: \n' ,'TC: ' ,tc,'\n TMC :', tmc,'\n FS: ', fs,'\n FC:' ,fc, '\n TS:' ,ts)
    FR = (tc) / (tc + tmc + fs)
    TVR = (tc) /(tc + tmc + fc)
    print('FR: ', FR)
    print('TVR: ', TVR)
    #print('FR: ', FR_l[:4])
    #print('TVR: ', TVR_l[:4])    
    return TVR, FR

def getMetrics_TVR_FR_V2(dict_valuesAB, dict_valuesBA, l_predictAB, l_predictBA):
    tc, tmc, fs, fc, ts = 0,0,0,0,0
    FR_l, TVR_l = [], []
    #dict_truth = copy.deepcopy(dict_values)
    dict_predictAB = copy.deepcopy(dict_valuesAB)
    dict_predictBA = copy.deepcopy(dict_valuesBA)

    for elem in l_predictAB:
        tup = tuple(elem)
        #print(tup)
        idpath, idname = os.path.split(tup[0])
        seqpath, seqname = os.path.split(idpath)
        l_id = idname.split('_')
        id_t = int( l_id[len(l_id)-1])
        seq_t = int(seqname)
        high , gt, rank, t_score = tup[1], tup[2], tup[3], tup[4] 
        try:
            dict_predictAB[id_t][seq_t] = (high, gt, rank, t_score) # en el id, con seq
        except:
            continue
    #print('dictionary of values predict: \n', dict_predictAB)

    for elem in l_predictBA:
        tup = tuple(elem)
        #print(tup)
        idpath, idname = os.path.split(tup[0])
        seqpath, seqname = os.path.split(idpath)
        l_id = idname.split('_')
        id_t = int( l_id[len(l_id)-1])
        seq_t = int(seqname)
        high , gt, rank, t_score = tup[1], tup[2], tup[3], tup[4] 
        try:
            dict_predictBA[id_t][seq_t] = (high, gt, rank, t_score) # en el id, con seq
        except:
            continue
    #print('dictionary of values predict: \n', dict_predictBA)
    
    ###### CALCULANDO ... TC, TMC, FS, FC, TS 
    for key_query_ID in dict_valuesAB:        
        #print('key: ', key_query_ID)
        for key_seq in dict_valuesAB[key_query_ID]:
            #tup_t = (bool(> thresh_score), GT, rank, true score)
            FR, TVR = 0, 0
            tup_t = dict_predictAB[int(key_query_ID)][key_seq]
            if(tup_t[1] == True):           # TRUES , if GT is True
                if(tup_t[0] == True and tup_t[2] > 0): # TC , score > threshold and (is in TOP10) 
                    tc = tc+1
                elif(tup_t[0] == True and tup_t[2] == 0): # TMC , score > threshold  and (is'n in TOP10)
                    tmc = tmc + 1
                elif(tup_t[0] == True and tup_t == -1):# Cuando no arroja NADA, cuento TMC
                    tmc = tmc + 1
                elif(tup_t[0] == False): #FS, score < threshold
                    fs = fs + 1
                    #AUMENTAR **** FR *****
                    #if(tc!= 0):
                    #    FR = (tc) / (tc + tmc + fs)
                    #    TVR = (tc) /(tc + tmc + fc)
                    #    if(FR != 0 and TVR != 0 and FR != 1 and TVR != 1):
                    #        FR_l.append(FR)
                    #        TVR_l.append(TVR)


                #elif(tup_t == -1):#cuando no arroja NADA
                    #continue

            elif(tup_t[1] == False) :    # FALSES , if GT is False
                if(tup_t[0] == True): # FC (score > threshol
                    fc = fc+1
                    #AUMENTAR **** TVR  ******
                    #if(tc!= 0):
                    #    FR = (tc) / (tc + tmc + fs)
                    #    TVR = (tc) /(tc + tmc + fc)
                    #    if(FR != 0 and TVR != 0 and FR != 1 and TVR != 1):
                    #        FR_l.append(FR)
                    #        TVR_l.append(TVR)
                elif(tup_t[0] == True and tup_t == -1):#Cuando no arroja NADA cuento FC
                    fc = fc+1
                elif(tup_t[0] == False): # TS score < threshold
                    ts = ts + 1
                #elif(tup_t == -1):
                    #continue       # VOID, reid no arrojo NADA
            
            #            
    ###### CALCULANDO ... TC, TMC, FS, FC, TS 
    
    for key_query_ID in dict_valuesBA:        
        #print('key: ', key_query_ID)
        for key_seq in dict_valuesBA[key_query_ID]:
            #tup_t = (bool(> thresh_score), GT, rank, true score)
            FR, TVR = 0, 0
            tup_t = dict_predictBA[int(key_query_ID)][key_seq]
            if(tup_t[1] == True):           # TRUES , if GT is True
                if(tup_t[0] == True and tup_t[2] > 0): # TC , score > threshold and (is in TOP10) 
                    tc = tc+1
                elif(tup_t[0] == True and tup_t[2] == 0): # TMC , score > threshold  and (is'n in TOP10)
                    tmc = tmc + 1
                elif(tup_t[0] == True and tup_t == -1):# Cuando no arroja NADA, cuento TMC
                    tmc = tmc + 1
                elif(tup_t[0] == False): #FS, score < threshold
                    fs = fs + 1
                
            elif(tup_t[1] == False) :    # FALSES , if GT is False
                if(tup_t[0] == True): # FC (score > threshol
                    fc = fc+1
                elif(tup_t[0] == True and tup_t == -1):#Cuando no arroja NADA cuento FC
                    fc = fc+1
                elif(tup_t[0] == False): # TS score < threshold
                    ts = ts + 1                
    
    #################################################################
    print('metrics: \n' ,'TC: ' ,tc,'\n TMC :', tmc,'\n FS: ', fs,'\n FC:' ,fc, '\n TS:' ,ts)
    FR = (tc) / (tc + tmc + fs)
    #TVR = (tc) /(tc + tmc + fc)
    
    try:
        TVR = (tc) /(tc + tmc + fc)
    except:
        TVR = 0
    ##print('FR: ', FR)
    #print('TVR: ', TVR)
    #print('FR: ', FR_l[:4])
    #print('TVR: ', TVR_l[:4])    
    
    return TVR, FR


def calculate_relations_new(dict_values, m_truth, l_predict):
    tc, tmc, fs, fc, ts = 0,0,0,0,0
    dict_truth = copy.deepcopy(dict_values)
    dict_predict = copy.deepcopy(dict_values)
   
    for elem in l_predict:
        tup = tuple(elem)
        #print(tup)
        idpath, idname = os.path.split(tup[1])
        seqpath, seqname = os.path.split(idpath)
        l_id = idname.split('_')
        id_t = int( l_id[len(l_id)-1])
        seq_t = int(seqname)
        high , gt, rank, t_score = tup[2], tup[3], tup[4], tup[5] 
        try:
            dict_predict[id_t][seq_t] = (high, gt, rank, t_score) # en el id, con seq
        except:
            continue
    print('dictionary of values predict: \n', dict_predict)
    
    ###### CALCULANDO ... TC, TMC, FS, FC, TS 
    for key_query_ID in dict_values:        
        #print('key: ', key_query_ID)
        for key_seq in dict_values[key_query_ID]:
            #tup_t = (bool(> thresh_score), GT, rank, true score)
            tup_t = dict_predict[int(key_query_ID)][key_seq]
            if(tup_t[1] == True):           # TRUES , if GT is True
                if(tup_t[0] == True and tup_t[2] > 0): # TC , score > threshold and (is in TOP10) 
                    tc = tc+1
                elif(tup_t[0] == True and tup_t[2] == 0): # TMC , score > threshold  and (is'n in TOP10)
                    tmc = tmc + 1
                elif(tup_t[0] == False): #FS, score < threshold
                    fs = fs + 1
                elif(tup_t == -1):
                    continue
            elif(tup_t[1] == False) :    # FALSES , if GT is False
                if(tup_t[0] == True): # FC (score > threshol
                    fc = fc+1
                elif(tup_t[0] == False): # TS score < threshold
                    ts = ts + 1
                elif(tup_t == -1):
                    continue       # VOID, reid no arrojo NADA
    return tc, tmc, fs, fc, ts


def generate_newResults(list_predict, beta, eta):
    new_predicts = []
    for tup in list_predict:
        t_tup = None
        if(tup[4] == -1):#VACIO
            t_tup = (tup[1], tup[2], tup[3], tup[4], tup[5], tup[6])
        elif(tup[4] == 0):# ISN'T ON VIDEO (PREDICTION)
            if(tup[6] >= float(beta)):
                t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
            else:
                t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
        elif(tup[4] > 0):# IS ON VIDEO (PREDICTION)
            if(tup[6] >= float(beta)):
                if(tup[4] <= int(eta)):
                    t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
                else:
                    t_tup = (tup[1], True, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
            else:
                if(tup[4] <= int(eta)):
                    t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
                else:
                    t_tup = (tup[1], False, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
        new_predicts.append(t_tup)
    return new_predicts

def main(argv=None):

    if FLAGS.mode == 'metrics':
        FLAGS.batch_size = 1
        print('**** METRICS ***')
        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        #end1, end2 = len(testAB) - 1, len(testBA) - 1
        predict_file_AB = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ab.csv'
        predict_file_BA = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ba.csv'
        print('predict file: ', predict_file_AB)
        #testAB = testAB[:end1]
        #testBA = testBA[:end2]
        #GENERANDO LISTA DE QUERYS        
        
        dict_values_AB = {}# aqui estaran dos los valores y resultados
        dict_values_BA = {}
        
        ##A->B
        matrix_truth_AB = []
        list_predict_AB = []
        
        #read predict
        predictFrameAB = pd.DataFrame()
        try:
            predictFrameAB = pd.read_csv(predict_file_AB, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_AB = predictFrameAB.to_numpy()
        #print('predict: \n', list_predict_AB)

        #read truth and generate values 0
        for carpet_seq in testAB:
            #querys names
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]
                
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0

                dict_values_AB[int(fname)] = dict_temp
            ##
            ground_truth = pd.DataFrame()
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            
            matrix_truth_AB.append( [ tuple(e) for e in ground_truth.to_numpy() ])# convierto a una matrix de lista de tuplas
        #print('truth: \n', matrix_truth_AB)
        
        #B->A
        matrix_truth_BA = []
        list_predict_BA = []

        predictFrameBA = pd.DataFrame()
        try:
            predictFrameBA = pd.read_csv(predict_file_BA, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_BA = predictFrameBA.to_numpy()
        #print('predict: \n', list_predict_BA)
        #read truth and generate values 0
        for carpet_seq in testBA:
            #querys names
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]
                
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0

                dict_values_BA[int(fname)] = dict_temp
            ##
            ground_truth = pd.DataFrame()
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            
            matrix_truth_BA.append( [ tuple(e) for e in ground_truth.to_numpy() ])# convierto a una matrix de lista de tuplas
        #print('truth: \n', matrix_truth_BA)
        
        print('dictionary of values ab: \n', dict_values_AB)
        print('dictionary of values ba: \n', dict_values_BA)
        
        tc1, tmc1, fs1, fc1, ts1 = calculate_relations_new(dict_values_AB, matrix_truth_AB, list_predict_AB)
        tc2, tmc2, fs2, fc2, ts2 = calculate_relations_new(dict_values_BA , matrix_truth_BA, list_predict_BA)
        tcT = tc1 + tc2 
        tmcT = tmc1 + tmc2
        fsT = fs1 + fs2
        fcT = fc1+ fc2
        tsT = ts1+ ts2
        print('metrics: \n' ,'TC: ' ,tcT,'\n TMC :', tmcT,'\n FS: ', fsT,'\n FC:' ,fcT, '\n TS:' ,tsT)
        FR = (tcT) / (tcT + tmcT + fsT)
        TVR = (tcT) /(tcT + tmcT + fcT)
        print('FR: ', FR)
        print('TVR: ', TVR)

        results_tup = [ ('TC','TMC', 'FS', 'FC', 'TS', 'FR', 'TVR'),\
                     (tcT, tmcT, fsT, fcT, tsT, FR, TVR) ]
        name_r = str(FLAGS.p_name).split('_')
        path_results = FLAGS.data_dir +'/results_'+name_r[2]+'_'+name_r[3]+'.csv'
        df = pd.DataFrame(np.array(results_tup))
        df.to_csv(path_results , header = False)
        return        

    if FLAGS.mode == 'classic_test':
        fpath, fname = os.path.split(FLAGS.cropps_path)
        fpath_reid = fpath + '/out_reid'
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        #reidentifier = personReID.personReIdentifier()
        reidentifier = personReIdentifier()#Second version Re-ID model using fast_reid
        topN = FLAGS.top
        print('*******CLASSIC RE-ID TEST*******')
        reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.cropps_path, fpath_reid, topN,show_query = True)

    if FLAGS.mode == 'rw_test':
        cropper = yolo_cropper.YOLOcropper()#objeto que recorta
        #reidentifier = personReID.personReIdentifier()#objeto que hace reid
        reidentifier = personReIdentifier()#Second version Re-ID model using fast_reid               
        print('*******REAL WORLD RE-ID SIMPLE TEST*******')
        ############# FLAGS
        fpath, fname = os.path.split(FLAGS.video_path )
        fpath_cropps = fpath + '/cropps'
        if not os.path.isfile(fpath_cropps):
            os.system('mkdir '+ fpath_cropps)
        
        tf.flags.DEFINE_string('out_cropps_path', fpath_cropps ,'out cropps path to gallery')
        
        cropper.personCropping(FLAGS.video_path, FLAGS.out_cropps_path)
        topN = FLAGS.top
        fpath_reid = fpath + '/out_reid'
        if not os.path.isfile(fpath_reid):
            os.system('mkdir '+ fpath_reid)
        ############### FIN FLAGS
        reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.out_cropps_path, fpath_reid, topN, show_query = True)
        #personReidentification(sess, FLAGS.query_path, FLAGS.out_cropps_path, fpath_reid, images, is_train, inference)
        
    if FLAGS.mode == 'data': #data_dir, dataset
        print('dataset y ReID')
        cropper = yolo_cropper.YOLOcropper()#objeto que recorta
        
        #reidentifier = personReID.personReIdentifier()#objeto que hace reid
        reidentifier = personReIdentifier()#Second version Re-ID model using fast_reid

        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        
        print(testAB)
        print(testBA)
        topN = 100
        t_frames_seq = int(FLAGS.t_skip)
        #in carpet test, we found , video.avi and queries /*.png
        for carpet_test_path in testAB:
            pReID_queries_per_videos(cropper, reidentifier, carpet_test_path , carpet_test_path + '/video_in.avi', topN, t_frames_seq)
        
        for carpet_test_path in testBA:
            pReID_queries_per_videos(cropper, reidentifier, carpet_test_path , carpet_test_path + '/video_in.avi', topN,t_frames_seq)
        #YOU SHOULD HAVE PREDICTS.CSV
    
    if FLAGS.mode == 'val':

        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))

        predict_values_AB = {}
        predict_values_AB = {}
        list_tup_predict_AB = []
        list_tup_predict_BA = []
        dict_truth_AB = {}
        dict_truth_BA = {}
        global key_rank
        # A -> B ****************************
        
        files_out_reid_AB = []
        for carpet_seq in testAB:
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                q_temp = []
                fpath, fname = os.path.split(query_path)
                l_q = fname.split('.')
                fname = l_q[0]
                l_q = fname.split('_')
                id_q = int(l_q[1])
                temp_dict = {}
                for sub_seq in sorted(glob.glob(fpath + '/seq_videos/*')):
                    q_temp.append(sub_seq +'/out_reid_'+ fname)
                    seq_path, seq_name = os.path.split(sub_seq)
                    temp_dict[int(seq_name)] = 0
                dict_truth_AB[id_q] = temp_dict
                files_out_reid_AB.append(q_temp)
                ## genero groundtruth y adiciono al dictionary
            ground_truth = pd.DataFrame()
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            m_truth = [ tuple(e) for e in ground_truth.to_numpy() ]
            for tup in m_truth:
                try:
                    dict_truth_AB[tup[1]][tup[9]] = 1
                except:
                    continue
            print("files_out_reid_AB:",files_out_reid_AB)
            print("dict_truth_AB:",dict_truth_AB)
            ###
            #files_out_reid_AB.append(q_temp)
        #files_out_reid_AB = np.sort(files_out_reid_AB)
        #print('queries: \n',files_out_reid_AB[0])
        #
        
        w=100
        h=200
        for file in files_out_reid_AB:
            #global key_rank
            for subfile in file:
                list_scores = pd.DataFrame()
                try:
                    list_scores = pd.read_csv(subfile+'/top/score_results.csv', header=None)
                except pd.errors.EmptyDataError:
                    print('el archivo CSV esta vacio\n')
                list_tup_scores = [ tuple(e) for e in list_scores.to_numpy() ]
                # ('path_sub', > thresh_score, GT, rank, true score)
                # generate id and seq
                idpath, idname = os.path.split(subfile)
                seqpath, seqname = os.path.split(idpath)
                l_id = idname.split('_')
                id_t = int( l_id[len(l_id)-1])
                seq_t = int(seqname)
                gt = dict_truth_AB[id_t][seq_t]

                print('key g: ', key_rank)
                if(key_rank == 'auto'):
                    print('entre')
                    if((list_tup_scores) and ( list_tup_scores[0][2] >= float(FLAGS.threshold))):
                        tup = (subfile, bool(1), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                        list_tup_predict_AB.append(tup)
                    elif(not list_tup_scores):#NEGATIVE
                        tup = (subfile, bool(0), bool(gt), -1, 0.0, 0.0) #NINGUN RANK , NI SCORE, PORQUE NO HUBO SALIDA
                        list_tup_predict_AB.append(tup)
                    else:
                        tup = (subfile, bool(0), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                        list_tup_predict_AB.append(tup)
                else:
                    #POSITIVE
                    if((list_tup_scores) and ( list_tup_scores[0][2] >= float(FLAGS.threshold))):
                        #print('rank 1:', list_tup_scores[0])
                        fig=plt.figure(figsize=(9, 9), num= idname +' ReID POSITIVE')
                        #plt.title('person ReIdentification')
                        count = 0
                        columns = 7
                        rows =3
                        i = 1
                        j = 1
                        
                        for i in range(1, columns*rows +1):
                            #print('file: ',subfile+'/top/'+ tup[1])
                            if(count > len(list_tup_scores)):
                                break
                            elif(count==0):
                                #tup = list_tup_scores[count-1]
                                p_temp = glob.glob(subfile+'/*.png')
                                print(p_temp)
                                image1 = cv2.imread(p_temp[0])
                                image1 = cv2.resize(image1, (w, h))
                                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                ax_temp = fig.add_subplot(rows, columns, i)
                                ax_temp.title.set_text('Query')
                                ax_temp.spines['bottom'].set_color('#00FF11')
                                ax_temp.spines['top'].set_color('#00FF11') 
                                ax_temp.spines['right'].set_color('#00FF11')
                                ax_temp.spines['left'].set_color('#00FF11')
                                ax_temp.spines['bottom'].set_linewidth(2)
                                ax_temp.spines['top'].set_linewidth(2) 
                                ax_temp.spines['right'].set_linewidth(2)
                                ax_temp.spines['left'].set_linewidth(2)
                                ax_temp.get_xaxis().set_visible(False)
                                ax_temp.get_yaxis().set_visible(False)
                                #plt.axis('off')
                                plt.imshow(image1)
                                count = count +1
                            else:
                                    tup = list_tup_scores[count-1]
                                    image1 = cv2.imread( subfile+'/top/'+str(count)+'_'+tup[1])
                                    #print(image1)
                                    image1 = cv2.resize(image1, (w, h))
                                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                    ax_temp = fig.add_subplot(rows, columns, i)
                                    ax_temp.spines['bottom'].set_color('#EB9133')
                                    ax_temp.spines['top'].set_color('#EB9133') 
                                    ax_temp.spines['right'].set_color('#EB9133')
                                    ax_temp.spines['left'].set_color('#EB9133')
                                    ax_temp.spines['bottom'].set_linewidth(2)
                                    ax_temp.spines['top'].set_linewidth(2) 
                                    ax_temp.spines['right'].set_linewidth(2)
                                    ax_temp.spines['left'].set_linewidth(2)
                                    ax_temp.title.set_text('Rank '+str(count))
                                    ax_temp.get_xaxis().set_visible(False)
                                    ax_temp.get_yaxis().set_visible(False)
                                    #plt.axis('off')
                                    plt.imshow(image1)
                                    count = count +1
                                                
                        axbox = plt.axes([0.6, 0.05, 0.2, 0.035])
                        text_box = TextBox(axbox, 'Insert number of rank if query appears ')
                        text_box.on_submit(submit_rank)

                        plt.show()                        
                        # 
                        #print('key g: ', key_rank)               
                        if(key_rank != '0' and key_rank != 'auto'):# IS on TOP10
                            print('key g: ', key_rank)
                            #rank = key_rank
                            tup = (subfile, bool(1), bool(gt), int(key_rank), list_tup_scores[int(key_rank)-1][2], list_tup_scores[0][2])
                            list_tup_predict_AB.append(tup)
                        elif(key_rank == '0'or key_rank == 'auto'):# ISN'T on TOP10
                            tup = (subfile, bool(1), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP y TOMO EL SCORE DEL RANK 1
                            list_tup_predict_AB.append(tup)
                    elif(not list_tup_scores):#NEGATIVE
                        #csv vacio, rank -1
                        print('negativo VACIO')
                        #(path, high >= 0.9, low < 0.9, rank_vacio, t_score, f_score)
                        tup = (subfile, bool(0), bool(gt), -1, 0.0, 0.0) #NINGUN RANK , NI SCORE, PORQUE NO HUBO SALIDA
                        list_tup_predict_AB.append(tup)
                    else:
                        print('negativo con DATOS')

                        fig=plt.figure(figsize=(9, 9), num= idname +' ReID NEGATIVE')
                        #plt.title('person ReIdentification')
                        count = 0
                        columns = 7
                        rows =3
                        i = 1
                        j = 1
                        
                        for i in range(1, columns*rows +1):
                            #print('file: ',subfile+'/top/'+ tup[1])
                            if(count > len(list_tup_scores)):
                                break
                            elif(count==0):
                                #tup = list_tup_scores[count-1]
                                p_temp = glob.glob(subfile+'/*.png')
                                print(p_temp)
                                image1 = cv2.imread(p_temp[0])
                                image1 = cv2.resize(image1, (w, h))
                                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                ax_temp = fig.add_subplot(rows, columns, i)
                                ax_temp.title.set_text('Query')
                                ax_temp.spines['bottom'].set_color('#00FF11')
                                ax_temp.spines['top'].set_color('#00FF11') 
                                ax_temp.spines['right'].set_color('#00FF11')
                                ax_temp.spines['left'].set_color('#00FF11')
                                ax_temp.spines['bottom'].set_linewidth(2)
                                ax_temp.spines['top'].set_linewidth(2) 
                                ax_temp.spines['right'].set_linewidth(2)
                                ax_temp.spines['left'].set_linewidth(2)
                                ax_temp.get_xaxis().set_visible(False)
                                ax_temp.get_yaxis().set_visible(False)
                                #plt.axis('off')
                                plt.imshow(image1)
                                count = count +1
                            else:
                                    tup = list_tup_scores[count-1]
                                    image1 = cv2.imread( subfile+'/top/'+str(count)+'_'+tup[1])
                                    #print(image1)
                                    image1 = cv2.resize(image1, (w, h))
                                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                    ax_temp = fig.add_subplot(rows, columns, i)
                                    ax_temp.spines['bottom'].set_color('#D50000')
                                    ax_temp.spines['top'].set_color('#D50000') 
                                    ax_temp.spines['right'].set_color('#D50000')
                                    ax_temp.spines['left'].set_color('#D50000')
                                    ax_temp.spines['bottom'].set_linewidth(2)
                                    ax_temp.spines['top'].set_linewidth(2) 
                                    ax_temp.spines['right'].set_linewidth(2)
                                    ax_temp.spines['left'].set_linewidth(2)
                                    ax_temp.title.set_text('Rank '+str(count))
                                    ax_temp.get_xaxis().set_visible(False)
                                    ax_temp.get_yaxis().set_visible(False)
                                    #plt.axis('off')
                                    plt.imshow(image1)
                                    count = count +1
                        
                        axbox = plt.axes([0.6, 0.05, 0.2, 0.035])
                        text_box = TextBox(axbox, 'Insert number of rank if query appears ')
                        text_box.on_submit(submit_rank)
                        plt.show()                        
                        # 
                        #print('key g: ', key_rank)               
                        if(key_rank != '0' and key_rank != 'auto'):# IS on TOP10
                            print('key g: ', key_rank)
                            #rank = key_rank
                            tup = (subfile, bool(0), bool(gt), int(key_rank), list_tup_scores[int(key_rank)-1][2], list_tup_scores[0][2])
                            list_tup_predict_AB.append(tup)
                        elif(key_rank == '0' or key_rank == 'auto'):# ISN'T on TOP10
                            tup = (subfile, bool(0), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                            list_tup_predict_AB.append(tup)
                print('added: ', list_tup_predict_AB[len(list_tup_predict_AB)-1])
                print('------------------------------------------------------------------------')
            #para volver a pegar el rank, en otra CONSULTA
            key_rank = ''
        
        df = pd.DataFrame(np.array(list_tup_predict_AB))
        df.to_csv(FLAGS.data_dir +'/'+ FLAGS.p_name +'_ab.csv', header = False)
        
        # B -> A
        files_out_reid_BA = []
        for carpet_seq in testBA:                
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                q_temp = []    
                #
                fpath, fname = os.path.split(query_path)
                l_q = fname.split('.')
                fname = l_q[0]
                l_q = fname.split('_')
                id_q = int(l_q[1])
                temp_dict = {}
                for sub_seq in sorted(glob.glob(fpath + '/seq_videos/*')):
                    q_temp.append(sub_seq +'/out_reid_'+ fname)
                    seq_path, seq_name = os.path.split(sub_seq)
                    temp_dict[int(seq_name)] = 0
                dict_truth_BA[id_q] = temp_dict
                files_out_reid_BA.append(q_temp)
            ## genero groundtruth y adiciono al dictionary
            ground_truth = pd.DataFrame()
            try:
                ground_truth = pd.read_csv(carpet_seq+'/ground_truth.csv', header=None)
            except pd.errors.EmptyDataError:
                print('el archivo CSV GROUND TRUTH esta vacio\n')
            m_truth = [ tuple(e) for e in ground_truth.to_numpy() ]
            for tup in m_truth:
                try:
                    dict_truth_BA[tup[1]][tup[9]] = 1
                except:
                    continue
            ###    
            #files_out_reid_BA.append(q_temp)
        #files_out_reid_BA = np.sort(files_out_reid_BA)
        #print('queries: \n',files_out_reid_BA[0])
        #            
        w=100
        h=200
        for file in files_out_reid_BA:
            #global key_rank
            for subfile in file:
                
                list_scores = pd.DataFrame()
                try:
                    list_scores = pd.read_csv(subfile+'/top/score_results.csv', header=None)
                except pd.errors.EmptyDataError:
                    print('el archivo CSV esta vacio\n')
                list_tup_scores = [ tuple(e) for e in list_scores.to_numpy() ]
                    # ('path_sub', > thresh_score, GT, rank, true score)
                # generate id and seq
                idpath, idname = os.path.split(subfile)
                seqpath, seqname = os.path.split(idpath)
                l_id = idname.split('_')
                id_t = int( l_id[len(l_id)-1])
                seq_t = int(seqname)
                gt = dict_truth_BA[id_t][seq_t]
                
                print('key g: ', key_rank)
                if(key_rank == 'auto'):
                    print('entre')
                    if((list_tup_scores) and ( list_tup_scores[0][2] >= float(FLAGS.threshold))):
                        tup = (subfile, bool(1), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                        list_tup_predict_BA.append(tup)
                    elif(not list_tup_scores):#NEGATIVE
                        tup = (subfile, bool(0), bool(gt), -1, 0.0, 0.0) #NINGUN RANK , NI SCORE, PORQUE NO HUBO SALIDA
                        list_tup_predict_BA.append(tup)
                    else:
                        tup = (subfile, bool(0), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                        list_tup_predict_BA.append(tup)
                else:
                    #POSITIVE
                    if((list_tup_scores) and ( list_tup_scores[0][2] >= float(FLAGS.threshold))):
                        #print('rank 1:', list_tup_scores[0])
                        fig=plt.figure(figsize=(9, 9), num=idname +' ReID POSITIVE')
                        #plt.title('person ReIdentification')
                        count = 0
                        columns = 7
                        rows =3
                        i = 1
                        j = 1
                        
                        for i in range(1, columns*rows +1):
                            #print('file: ',subfile+'/top/'+ tup[1])
                            if(count > len(list_tup_scores)):
                                break
                            elif(count==0):
                                #tup = list_tup_scores[count-1]
                                p_temp = glob.glob(subfile+'/*.png')
                                print(p_temp)
                                image1 = cv2.imread(p_temp[0])
                                image1 = cv2.resize(image1, (w, h))
                                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                ax_temp = fig.add_subplot(rows, columns, i)
                                ax_temp.title.set_text('Query')
                                ax_temp.spines['bottom'].set_color('#00FF11')
                                ax_temp.spines['top'].set_color('#00FF11') 
                                ax_temp.spines['right'].set_color('#00FF11')
                                ax_temp.spines['left'].set_color('#00FF11')
                                ax_temp.spines['bottom'].set_linewidth(2)
                                ax_temp.spines['top'].set_linewidth(2) 
                                ax_temp.spines['right'].set_linewidth(2)
                                ax_temp.spines['left'].set_linewidth(2)
                                ax_temp.get_xaxis().set_visible(False)
                                ax_temp.get_yaxis().set_visible(False)
                                #plt.axis('off')
                                plt.imshow(image1)
                                count = count +1
                            else:
                                    tup = list_tup_scores[count-1]
                                    image1 = cv2.imread( subfile+'/top/'+str(count)+'_'+tup[1])
                                    #print(image1)
                                    image1 = cv2.resize(image1, (w, h))
                                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                    ax_temp = fig.add_subplot(rows, columns, i)
                                    ax_temp.spines['bottom'].set_color('#EB9133')
                                    ax_temp.spines['top'].set_color('#EB9133') 
                                    ax_temp.spines['right'].set_color('#EB9133')
                                    ax_temp.spines['left'].set_color('#EB9133')
                                    ax_temp.spines['bottom'].set_linewidth(2)
                                    ax_temp.spines['top'].set_linewidth(2) 
                                    ax_temp.spines['right'].set_linewidth(2)
                                    ax_temp.spines['left'].set_linewidth(2)
                                    ax_temp.title.set_text('Rank '+str(count))
                                    ax_temp.get_xaxis().set_visible(False)
                                    ax_temp.get_yaxis().set_visible(False)
                                    #plt.axis('off')
                                    plt.imshow(image1)
                                    count = count +1
                        axbox = plt.axes([0.6, 0.05, 0.2, 0.035])
                        text_box = TextBox(axbox, 'Insert number of rank if query appears ')
                        text_box.on_submit(submit_rank)
                        plt.show()                        
                        # 
                                    
                        if(key_rank != '0' and key_rank != 'auto'):# IS on TOP10
                            print('key g: ', key_rank)
                            #rank = key_rank
                            tup = (subfile, bool(1), bool(gt), int(key_rank), list_tup_scores[int(key_rank)-1][2], list_tup_scores[0][2])
                            list_tup_predict_BA.append(tup)
                        elif(key_rank == '0' or key_rank == 'auto'):# ISN'T on TOP10
                            tup = (subfile, bool(1), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                            list_tup_predict_BA.append(tup)
                    elif(not list_tup_scores):#NEGATIVE
                        #csv vacio, rank -1
                        print('negativo VACIO')
                        #(path, high >= 0.9, low < 0.9, rank_vacio, t_score, f_score)
                        tup = (subfile, bool(0), bool(gt), -1, 0.0, 0.0) #NINGUN RANK , NI SCORE, PORQUE NO HUBO SALIDA
                        list_tup_predict_BA.append(tup)
                    else:
                        print('negativo con DATOS')

                        fig=plt.figure(figsize=(9, 9), num= idname +' ReID NEGATIVE')
                        #plt.title('person ReIdentification')
                        count = 0
                        columns = 7
                        rows =3
                        i = 1
                        j = 1
                        
                        for i in range(1, columns*rows +1):
                            #print('file: ',subfile+'/top/'+ tup[1])
                            if(count > len(list_tup_scores)):
                                break
                            elif(count==0):
                                #tup = list_tup_scores[count-1]
                                p_temp = glob.glob(subfile+'/*.png')
                                print(p_temp)
                                image1 = cv2.imread(p_temp[0])
                                image1 = cv2.resize(image1, (w, h))
                                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                ax_temp = fig.add_subplot(rows, columns, i)
                                ax_temp.title.set_text('Query')
                                ax_temp.spines['bottom'].set_color('#00FF11')
                                ax_temp.spines['top'].set_color('#00FF11') 
                                ax_temp.spines['right'].set_color('#00FF11')
                                ax_temp.spines['left'].set_color('#00FF11')
                                ax_temp.spines['bottom'].set_linewidth(2)
                                ax_temp.spines['top'].set_linewidth(2) 
                                ax_temp.spines['right'].set_linewidth(2)
                                ax_temp.spines['left'].set_linewidth(2)
                                ax_temp.get_xaxis().set_visible(False)
                                ax_temp.get_yaxis().set_visible(False)
                                #plt.axis('off')
                                plt.imshow(image1)
                                count = count +1
                            else:
                                    tup = list_tup_scores[count-1]
                                    image1 = cv2.imread( subfile+'/top/'+str(count)+'_'+tup[1])
                                    #print(image1)
                                    image1 = cv2.resize(image1, (w, h))
                                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                                    ax_temp = fig.add_subplot(rows, columns, i)
                                    ax_temp.spines['bottom'].set_color('#D50000')
                                    ax_temp.spines['top'].set_color('#D50000') 
                                    ax_temp.spines['right'].set_color('#D50000')
                                    ax_temp.spines['left'].set_color('#D50000')
                                    ax_temp.spines['bottom'].set_linewidth(2)
                                    ax_temp.spines['top'].set_linewidth(2) 
                                    ax_temp.spines['right'].set_linewidth(2)
                                    ax_temp.spines['left'].set_linewidth(2)
                                    ax_temp.title.set_text('Rank '+str(count))
                                    ax_temp.get_xaxis().set_visible(False)
                                    ax_temp.get_yaxis().set_visible(False)
                                    #plt.axis('off')
                                    plt.imshow(image1)
                                    count = count +1
                        axbox = plt.axes([0.6, 0.05, 0.2, 0.035])
                        text_box = TextBox(axbox, 'Insert number of rank if query appears ')
                        text_box.on_submit(submit_rank)
                        plt.show()                        
                        # 
                        #print('key g: ', key_rank)               
                        if(key_rank != '0' and key_rank != 'auto'):# IS on TOP10
                            #print('key g: ', key_rank)
                            #rank = key_rank
                            tup = (subfile, bool(0), bool(gt), int(key_rank), list_tup_scores[int(key_rank)-1][2], list_tup_scores[0][2])
                            list_tup_predict_BA.append(tup)
                        elif(key_rank == '0' or key_rank == 'auto'):# ISN'T on TOP10
                            tup = (subfile, bool(0), bool(gt), 0, list_tup_scores[0][2], list_tup_scores[0][2])# rank 0, porq no esta en el TOP
                            list_tup_predict_BA.append(tup)
                                        
                print('added: ', list_tup_predict_BA[len(list_tup_predict_BA)-1])
                print('------------------------------------------------------------------------')
            #para volver a pegar el rank, en otra CONSULTA
            key_rank = ''
        df = pd.DataFrame(np.array(list_tup_predict_BA))
        df.to_csv(FLAGS.data_dir +'/'+ FLAGS.p_name +'_ba.csv', header = False)
    
    if FLAGS.mode == 'val2':

        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        #end1, end2 = len(testAB) - 1, len(testBA) - 1
        predict_file_AB = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ab.csv'
        predict_file_BA = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ba.csv'
        #print('predict file: ', predict_file_AB)
        print('beta: ' , FLAGS.beta, ' eta: ', FLAGS.eta)
        #A-B
        #read predict
        predictFrameAB = pd.DataFrame()
        try:
            predictFrameAB = pd.read_csv(predict_file_AB, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_AB = [ tuple(e) for e in predictFrameAB.to_numpy() ]
        #( , path, > beta, gt, rank, score_rank, score_top1)
        #print('predict: \n', list_predict_AB)
        new_predicts_AB = []
        for tup in list_predict_AB:
            t_tup = None
            if(tup[4] == -1):#VACIO
                t_tup = (tup[1], tup[2], tup[3], tup[4], tup[5], tup[6])
            elif(tup[4] == 0):# ISN'T ON VIDEO (PREDICTION)
                if(tup[6] >= float(FLAGS.beta)):
                    t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
                else:
                    t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
            elif(tup[4] > 0):# IS ON VIDEO (PREDICTION)
                if(tup[6] >= float(FLAGS.beta)):
                    if(tup[4] <= int(FLAGS.eta)):
                        t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
                    else:
                        t_tup = (tup[1], True, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
                else:
                    if(tup[4] <= int(FLAGS.eta)):
                        t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
                    else:
                        t_tup = (tup[1], False, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
            new_predicts_AB.append(t_tup)
        #
        path_new_AB = FLAGS.data_dir +'/'+ \
            FLAGS.p_name+'_'+str(FLAGS.beta)+'_'+str(FLAGS.eta)+'_ab.csv'
        df = pd.DataFrame(np.array(new_predicts_AB))
        df.to_csv(path_new_AB , header = False)

        ## B-A
            #read predict
        predictFrameBA = pd.DataFrame()
        try:
            predictFrameBA = pd.read_csv(predict_file_BA, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_BA = [ tuple(e) for e in predictFrameBA.to_numpy() ]
        #( , path, > beta, gt, rank, score_rank, score_top1)
        #print('predict: \n', list_predict_BA)
        new_predicts_BA = []
        for tup in list_predict_BA:
            t_tup = None
            if(tup[4] == -1):#VACIO
                t_tup = (tup[1], tup[2], tup[3], tup[4], tup[5], tup[6])
            elif(tup[4] == 0):# ISN'T ON VIDEO (PREDICTION)
                if(tup[6] >= float(FLAGS.beta)):
                    t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
                else:
                    t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
            elif(tup[4] > 0):# IS ON VIDEO (PREDICTION)
                if(tup[6] >= float(FLAGS.beta)):
                    if(tup[4] <= int(FLAGS.eta)):
                        t_tup = (tup[1], True, tup[3], tup[4], tup[5], tup[6])
                    else:
                        t_tup = (tup[1], True, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
                else:
                    if(tup[4] <= int(FLAGS.eta)):
                        t_tup = (tup[1], False, tup[3], tup[4], tup[5], tup[6])
                    else:
                        t_tup = (tup[1], False, tup[3], 0, tup[6], tup[6])#ISN'T ON TOP
            new_predicts_BA.append(t_tup)
        #
        path_new_BA = FLAGS.data_dir +'/'+ \
            FLAGS.p_name+'_'+str(FLAGS.beta)+'_'+str(FLAGS.eta)+'_ba.csv'
        df = pd.DataFrame(np.array(new_predicts_BA))
        df.to_csv(path_new_BA , header = False)
        ####
        
    if FLAGS.mode == 'graph':
        #testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        #testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        #end1, end2 = len(testAB) - 1, len(testBA) - 1
        #beta_list = np.linspace(0.0, 1.0, num=20)
        beta_list = np.arange(0.0, 1.0, 0.02)
        #print(beta_list)
        predict_file_AB = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ab.csv'
        predict_file_BA = FLAGS.data_dir +'/'+ FLAGS.p_name +'_ba.csv'
                    
        dict_graph = {}
        dict_valuesAB = {}
        dict_valuesBA = {}
        
        #read predict AB
        predictFrameAB = pd.DataFrame()
        try:
            predictFrameAB = pd.read_csv(predict_file_AB, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_AB = [ tuple(e) for e in predictFrameAB.to_numpy() ]
        
        #read predict BA
        predictFrameBA = pd.DataFrame()
        try:
            predictFrameBA = pd.read_csv(predict_file_BA, header=None)
        except pd.errors.EmptyDataError:
            print('el archivo CSV PREDICT esta vacio\n')
        list_predict_BA = [ tuple(e) for e in predictFrameBA.to_numpy() ]

        testAB = sorted(glob.glob(FLAGS.data_dir + '/A-B/*'))
        testBA = sorted(glob.glob(FLAGS.data_dir + '/B-A/*'))
        
        for carpet_seq in testAB:
            #querys names
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]                    
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0
                dict_valuesAB[int(fname)] = dict_temp
            ##            
        for carpet_seq in testBA:
            #querys names
            for query_path in sorted(glob.glob(carpet_seq + '/*.png')):
                fpath, fname = os.path.split(query_path)
                fname = fname.split('_')[1]
                fname = fname.split('.')[0]                    
                #print(fpath)
                dict_temp = {}
                for seq_path in sorted(glob.glob(fpath + '/seq_videos/*')):
                    seq_path_temp, seq_name = os.path.split(seq_path)
                    #print('seq: ', seq_name)
                    dict_temp[int(seq_name)] = 0
                dict_valuesBA[int(fname)] = dict_temp
        results_TVR = []
        results_FR = []
        tops = [1,10,20]
        for t in tops:
            temp_tvr = []
            temp_fr = []
            for beta in beta_list:
                new_l_ab = generate_newResults(list_predict_AB, beta, t)
                new_l_ba = generate_newResults(list_predict_BA, beta, t)
                TVR, FR = getMetrics_TVR_FR_V2(dict_valuesAB, dict_valuesBA, \
                                new_l_ab, new_l_ba)
                temp_tvr.append(TVR)
                temp_fr.append(FR)
            results_TVR.append(temp_tvr)
            results_FR.append(temp_fr)
        print('shape tvr: ',np.shape(results_TVR))
        #beta_list = [str(e) for e in beta_list]
        #tops = [str(e) for e in tops]
        print(beta_list)
        #***
        #save tvr and fr in .CSV
        df = pd.DataFrame(np.around(results_TVR, decimals = 3).transpose() , \
            columns = tops, \
            index = np.around(beta_list, decimals = 2))
        df.to_csv(FLAGS.data_dir + '/tvrResults.csv' , header = True, index=True)

        df2 = pd.DataFrame(np.around(results_FR, decimals = 2).transpose(), \
            columns = tops, \
            index = np.around(beta_list, decimals = 2))
        df2.to_csv(FLAGS.data_dir + '/frResults.csv' , header = True, index=True)
        
        ########### Alpha ###################
        #print("Shape results_TVR: ",np.shape((results_TVR)))
        #print("results_TVR:",results_TVR)
        #print("Shape transpose results_TVR: ",np.shape(np.around(results_TVR, decimals = 3).transpose()))
        #print(" transpose results_TVR:",np.around(results_TVR, decimals = 3).transpose())
        #
        #print("tops:",tops)
        #print("beta_list:",beta_list)
        
        #Score = alpha * FR + (1-alpha) * TVR.
        alpha_list = [0.5,1,2]
        tvr = np.around(results_TVR, decimals = 3).transpose()
        fr = np.around(results_FR, decimals = 2).transpose()
        #aplha_table=np.zeros_like(tvr)
        #s_tvr=np.shape(tvr)
        print("alpha_list:",alpha_list)
        print("np.shape tvr:",np.shape(tvr))
        print("np.shape fr",np.shape(fr))

        
        for c in range(3):
            aplha_table=np.zeros_like(tvr)
            for r in range(len(tvr)):
                for a in range(len(alpha_list)):
                    #First alpha score
                    #Score = alpha * FR + (1-alpha) * TVR.
                    #aplha_table[r][a]=(alpha_list[a]*fr[r][c])+(1.0-alpha_list[a])*tvr[r][c]
                    
                    #Seconda alpha score
                    #Score = alpha * FR + (1-alpha) * TVR.
                    #F-alpha = (1+alpha^2) * ( (TVR * FR) / (alpha^2* TVR) + FR)
                    aplha_table[r][a]=(1+pow(alpha_list[a],2))*(tvr[r][c]*fr[r][c]) / ((pow(alpha_list[a],2)*tvr[r][c])+fr[r][c])
            print("aplha_table:",aplha_table)
            df2 = pd.DataFrame(np.around(aplha_table, decimals = 2), \
            columns = alpha_list, \
            index = np.around(beta_list, decimals = 2))
            df2.to_csv(FLAGS.data_dir + '/alphaResults_eta_'+str(tops[c])+'_.csv' , header = True, index=True)
            #table_alpha_list.append(aplha_table)
                
       # print("table_aplha_table:",len(table_alpha_list))

        #for y in range(len(table_alpha_list)):
        #   for x in range(len(table_alpha_list[0])):
                

        #####################################

        #***
        fig, ax1 = plt.subplots(figsize=(9, 6), num = ' Results for TVR and FR')
        
        #plt.title('τ = '+str(FLAGS.t_skip))
        
        cont = 0
        color = '#3366ff'
        lns1 = []
        for elem in results_FR:
            #print('ELEM2: ',elem)
            if(cont==2): mk='*-'
            elif(cont==1): mk ='o-'
            else: mk = '^-'
            t_lns = ax1.plot(beta_list, elem , mk, color=color, label='η = '+str(tops[cont]),\
                    lw=2, alpha=0.7,markersize=10)
            lns1.append(t_lns)
            cont=cont+1
        ax1.set_xlabel('Threshold β',fontsize='xx-large')
        ax1.grid()
        color = 'tab:blue'
        ax1.set_ylabel('Finding Rate', color=color,fontsize='xx-large')
        ax1.tick_params(axis='y', labelcolor=color)

        #ax1.legend(loc='middle left', shadow=True, fontsize='small')
        #**
        ax2 = ax1.twinx()
        cont = 0
        color = '#C00000'
        lns2 = []
        for elem in results_TVR:
            #print('ELEM: ',elem)
            if(cont==2): mk='*-'
            elif(cont==1): mk ='o-'
            else: mk = '^-'
            t_lns = ax2.plot(beta_list, elem , mk,color=color, label='η = '+str(tops[cont]),\
                lw=2, alpha=0.7, markersize=10)
            lns2.append(t_lns)
            cont=cont+1
        color = 'tab:red'
        ax2.set_ylabel('True Validation Rate', color=color,fontsize='xx-large')  # we already handled the x-label with ax1
        #ax2.plot(t, data2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        tup_l = [tuple(e) for e in zip(np.ravel(lns1),np.ravel(lns2))]
        #print(tup_l)
        labs = [l.get_label() for l in np.ravel(lns1)]
        ax1.legend(tup_l,labs, loc='lower left', shadow=True, fontsize='small',\
            handler_map={tuple: HandlerTuple(ndivide=None)},\
            bbox_to_anchor=(0., 1.02, 1., .102), ncol= len(labs),\
            mode="expand", borderaxespad=0. , prop={'size': 11})
        #for tup in tup_l:
        #    ax1.legend([tup], [])
        #legend = 
        #print('lns: ',np.ravel(lns))
        #lns = np.copy(np.ravel(lns))
        #labs = [l.get_label() for l in lns]
        #for e in lns:
            #e.set_color('black')
            #print('color: ',e[0])
        #print('labels: ',lns)
        #ax1.legend(lns,labs, loc='middle left', shadow=True, fontsize='small')
        #ax2.legend(loc='middle left', shadow=True, fontsize='small')
        #plt.xlabel('Threshold β')
        #plt.ylabel('True Validation Rate')
        #plt.legend()
        plt.savefig(FLAGS.data_dir +'/result_FR_TVR.png', dpi=150)
        plt.show()
        #*************************************************************

        
if __name__ == '__main__':
    tf.compat.v1.app.run()