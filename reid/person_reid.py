import argparse
import logging
import sys

import numpy as np
import torch

import os
import cv2
import pandas as pd
import glob 

#from torch.backends import cudnn

sys.path.append('.')

#from fastreid.evaluation import evaluate_rank
from reid.fastreid.config import get_cfg
from reid.fastreid.utils.logger import setup_logger
from reid.fastreid.data import build_reid_test_loader2
from reid.demo.predictor import FeatureExtractionDemo
from reid.fastreid.utils.visualizer import Visualizer

logger = logging.getLogger('fastreid.visualize_result')
setup_logger(name="fastreid")

def setup_cfg(config_file,opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    #print("config_file:", config_file)
    
    cfg.merge_from_file(config_file)
    #print("HERRE===================================")
    cfg.merge_from_list(opts)
    
    #print("args.opts: ",args.opts)
    cfg.freeze()
    return cfg


class personReIdentifier(object):
    def __init__(self):
        self.path_dataset='/home/josemiki/FF-PRID-2020/reid/video1'
        self.dataset_name='MyMarket'
        self.topN=5
        ###################### DUKE ########################################
        #self.cfg_path='reid/logs/dukemtmc/agw_R50/config.yaml'
        #self.weights_path='reid/logs/dukemtmc/agw_R50/model_final.pth'
        #self.cfg_path='reid/logs/dukemtmc/bagtricks_R50/config.yaml'
        #self.weights_path='reid/logs/dukemtmc/bagtricks_R50/model_final.pth'
        #self.cfg_path='reid/logs/dukemtmc/sbs_R50/config.yaml'
        #self.weights_path='reid/logs/dukemtmc/sbs_R50/model_final.pth'
        ##Warning
        ##self.cfg_path='reid/logs/dukemtmc/mgn_R50-ibn/config.yaml'
        ##self.weights_path='reid/logs/dukemtmc/mgn_R50-ibn/model_final.pth'
        
        ####################### MARKET #####################################
        #self.cfg_path='reid/logs/market1501/agw_R50/config.yaml'
        #self.weights_path='reid/logs/market1501/agw_R50/model_final.pth'
        ##Warning
        ##self.cfg_path='reid/logs/market1501/bagtricks_R50/config.yaml'
        ##self.weights_path='reid/logs/market1501/bagtricks_R50/model_final.pth'
        #self.cfg_path='reid/logs/market1501/sbs_R50/config.yaml'
        #self.weights_path='reid/logs/market1501/sbs_R50/model_final.pth'
        #self.cfg_path='reid/logs/market1501/mgn_R50-ibn/config.yaml'
        #self.weights_path='reid/logs/market1501/mgn_R50-ibn/model_final.pth'

        ####################### CUHK #####################################
        #self.cfg_path='reid/logs/cuhk03/agw_R50/config.yaml'
        #self.weights_path='reid/logs/cuhk03/agw_R50/model_final.pth'
        #self.cfg_path='reid/logs/cuhk03/bagtricks_R50/config.yaml'
        #self.weights_path='reid/logs/cuhk03/bagtricks_R50/model_final.pth'
        #self.cfg_path='reid/logs/cuhk03/sbs_R50/config.yaml'
        #self.weights_path='reid/logs/cuhk03/sbs_R50/model_final.pth'
        #self.cfg_path='reid/logs/cuhk03/mgn_R50-ibn/config.yaml'
        #self.weights_path='reid/logs/cuhk03/mgn_R50-ibn/model_final.pth'

        ####################### MARKET + DUKE + CUHK #####################################
        #self.cfg_path='reid/logs/market+duke+cuhk/agw_R50/config.yaml'
        #self.weights_path='reid/logs/market+duke+cuhk/agw_R50/model_final.pth'
        #self.cfg_path='reid/logs/market+duke+cuhk/bagtricks_R50/config.yaml'
        #self.weights_path='reid/logs/market+duke+cuhk/bagtricks_R50/model_final.pth'
        #self.cfg_path='reid/logs/market+duke+cuhk/sbs_R50/config.yaml'
        #self.weights_path='reid/logs/market+duke+cuhk/sbs_R50/model_final.pth'
        #self.cfg_path='reid/logs/market+duke+cuhk/mgn_R50-ibn/config.yaml'
        #self.weights_path='reid/logs/market+duke+cuhk/mgn_R50-ibn/model_final.pth'

        ####################### SCALED MARKET + DUKE + CUHK #####################################
        #self.cfg_path='reid/logs/scaled_market+duke+cuhk/agw_R50/config.yaml'
        #self.weights_path='reid/logs/scaled_market+duke+cuhk/agw_R50/model_final.pth'
        #self.cfg_path='reid/logs/scaled_market+duke+cuhk/bagtricks_R50/config.yaml'
        #self.weights_path='reid/logs/scaled_market+duke+cuhk/bagtricks_R50/model_final.pth'
        self.cfg_path='reid/logs/scaled_market+duke+cuhk/sbs_R50/config.yaml'
        self.weights_path='reid/logs/scaled_market+duke+cuhk/sbs_R50/model_final.pth'
        #self.cfg_path='reid/logs/scaled_market+duke+cuhk/mgn_R50-ibn/config.yaml'
        #self.weights_path='reid/logs/scaled_market+duke+cuhk/mgn_R50-ibn/model_final.pth'

    def PersonReIdentification(self, cropps_path, topN, v,queries_path):
        self.topN=topN
        self.path_dataset=cropps_path
        cropps= os.listdir(cropps_path+"/cropps")
        if len(cropps)>0:
            print("======================> self.cfg_path:",self.cfg_path)
            print("======================> self.weights_path:",self.weights_path)
            cfg = setup_cfg(self.cfg_path,['MODEL.WEIGHTS', self.weights_path])
            #print("##############################3")
            test_loader, num_query, gallery= build_reid_test_loader2(cfg, self.dataset_name, self.path_dataset)
            demo = FeatureExtractionDemo(cfg, parallel=False)
            #print("=================Start extracting image features==============")
            feats = []
            pids = []
            camids = []
            for (feat, pid, camid) in demo.run_on_loader(test_loader):#, total=len(test_loader)):
                feats.append(feat)
                pids.extend(pid)
                camids.extend(camid)

            feats = torch.cat(feats, dim=0)
            q_feat = feats[:num_query]
            g_feat = feats[num_query:]
            # compute cosine distance
            distmat = 1 - torch.mm(q_feat, g_feat.t())
            distmat = distmat.numpy()
            '''
            print("size gallery: ",len(gallery))
            print("gallery[0]: ",gallery[0])
            print("gallery[1]: ",gallery[1])
            
            for q in range(num_query):
                flat_dist_mat=distmat[q].flatten().tolist()
                #print("flat_dist_mat:",flat_dist_mat)
                sorted_lis=sorted(flat_dist_mat)
                rank_n=sorted_lis[:topN]
                #print("rank_n:",rank_n)
                for x in rank_n:
                    idx=flat_dist_mat.index(x)
                    print("flat_dist_mat:",flat_dist_mat[idx])
                    print(gallery[idx+1+q][0])
                print()
            '''
            queries=[x[0] for x in gallery[:num_query]]#gallery[:num_query]
            
            #print("queries:",queries)

            for q in queries:
                qpath, qname = os.path.split(q)
                temp_qname = qname
                qname = qname.split('.')

                fpath_reid = v + '/out_reid_'+ qname[0]
                if not os.path.isfile(fpath_reid):
                    os.system('mkdir '+ fpath_reid)
                ## guardo la query
                temp_query = cv2.imread(q)
                cv2.imwrite(fpath_reid+'/'+temp_qname ,temp_query)
                ##
                fpath_reid_out = fpath_reid +'/top'
                if not os.path.isfile(fpath_reid_out):
                    os.system('mkdir '+ fpath_reid_out)
                #reidentifier.PersonReIdentification(q, fpath_cropps, fpath_reid_out, 
                # topN, show_query = False)

                list_all=[]
                list_reid_coords=[]
                list_score=[]

                id_q=queries.index(q)
                #print("id_q:",id_q)
                flat_dist_mat=distmat[id_q].flatten().tolist()
                #print("======flat_dist_mat:",flat_dist_mat)
                #print("======pid:", pid )
                sorted_lis=sorted(flat_dist_mat)
                rank_n=sorted_lis[:topN]
                #print("======rank_n:",rank_n)
                #print("$$$$$Gallery: ",gallery)

                for x in rank_n:
                    idx=flat_dist_mat.index(x)
                    #print("idx:",idx)
                    #print("gallery[idx+len(q_feat)]:",gallery[idx+len(q_feat)])
                    #print("gallery[idx+1+id_q]:",gallery[idx+1+id_q
                    g_name=gallery[idx+len(q_feat)][0]
                    #print(">>>>>>>",g_name)
                    #print("flat_dist_mat[idx]: ",flat_dist_mat[idx])
                    tup=(g_name,flat_dist_mat[idx],flat_dist_mat[idx])
                    list_all.append(tup)

                #print("###List All")
                #print(list_all)

                i=0
                for e in list_all:
                    temp_img = cv2.imread(e[0])
                    # estoy leyendo el crop de disco ... para luego escribir en otro lado
                    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                    fpath, fname = os.path.split(e[0])
                    if (i > topN ):
                        break
                    #estoy escribiendo ..
                    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)                
                    #CARPETA DONDE SE ESCRIBIRA LA SALIDA, LOS RESULTADOS
                    cv2.imwrite(fpath_reid_out+'/'+str(i+1)+'_'+fname, temp_img)
                    path_f, name_f = os.path.split(e[0])
                    splits_coords = name_f.rsplit('_')
                    #print("################")
                    #print("splits_coords:",splits_coords)
                    last_coord = splits_coords[5].rsplit('.')
                    #print("################")
                    i = i +1
                    list_reid_coords.append(( int(splits_coords[1]), splits_coords[2], splits_coords[3], splits_coords[4], last_coord[0]))
                    #pathi, nameimage = e[0]
                    list_score.append((name_f, 1.0-e[1], e[2]))
                    #print (i, e[0]," - ", e[1], " - ", e[2])
                ## escribo un csv
                df = pd.DataFrame(np.array(list_reid_coords))
                df.to_csv(fpath_reid_out+"/coords_results.csv", header = False)
                df = pd.DataFrame(np.array(list_score))
                df.to_csv(fpath_reid_out+"/score_results.csv", header = False)
            del test_loader, num_query, gallery#= build_reid_test_loader2(cfg, self.dataset_name, self.path_dataset)
            del demo
        else:
            #print("queries_path: ",queries_path)
            
            queries=sorted(glob.glob( queries_path+'/*.png')) #[x for x in queries_path]
            #print(queries)
            #for q in queries:
            #    print("q:",q)
            #    print("idx: ",queries.index(q))

            for q in queries:
                #print("Here=========================")
                #print("q:",q)
                qpath, qname = os.path.split(q)
                temp_qname = qname
                qname = qname.split('.')

                fpath_reid = v + '/out_reid_'+ qname[0]
                if not os.path.isfile(fpath_reid):
                    os.system('mkdir '+ fpath_reid)
                ## guardo la query
                temp_query = cv2.imread(q)
                cv2.imwrite(fpath_reid+'/'+temp_qname ,temp_query)
                ##
                fpath_reid_out = fpath_reid +'/top'
                if not os.path.isfile(fpath_reid_out):
                    os.system('mkdir '+ fpath_reid_out)
                
                list_all=[]
                list_reid_coords=[]
                list_score=[]
                ## escribo un csv
                df = pd.DataFrame(np.array(list_reid_coords))
                df.to_csv(fpath_reid_out+"/coords_results.csv", header = False)
                df = pd.DataFrame(np.array(list_score))
                df.to_csv(fpath_reid_out+"/score_results.csv", header = False)
            #del test_loader, num_query, gallery#= build_reid_test_loader2(cfg, self.dataset_name, self.path_dataset)
            #del demo
        torch.cuda.empty_cache()
        




if __name__ == '__main__':
    path_dataset='/home/josemiki/FF-PRID-2020/fast-reid/video1'
    reid = personReIdentifier()
    reid.PersonReIdentification(path_dataset,4)
    