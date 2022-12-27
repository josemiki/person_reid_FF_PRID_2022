import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
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
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    #print("args.opts: ",args.opts)
    cfg.freeze()
    return cfg


class personReIdentifier(object):
    def __init__(self):
        #cudnn.benchmark = True
        #self.
        #self.
        #self.cfg_path='reid/logs/dukemtmc/agw_R50/config.yaml'
        #self.weights_path='reid/logs/dukemtmc/agw_R50/model_final.pth'
        self.path_dataset='/home/josemiki/FF-PRID-2020/reid/video1'
        self.dataset_name='MyMarket'
        self.topN=5
        self.cfg_path='reid/logs/market1501/agw_R50/config.yaml'
        self.weights_path='reid/logs/market1501/agw_R50/model_final.pth'

    def PersonReIdentification(self, cropps_path, topN):
        self.topN=topN
        self.path_dataset=cropps_path
        cfg = setup_cfg(self.cfg_path,['MODEL.WEIGHTS', self.weights_path])
        test_loader, num_query ,gallery= build_reid_test_loader2(cfg, self.dataset_name, self.path_dataset)
        demo = FeatureExtractionDemo(cfg, parallel=True)
        #print("Start extracting image features")
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

        print("size gallery: ",len(gallery))
        for q in range(len(q_feat)):
            flat_dist_mat=distmat[q].flatten().tolist()
            sorted_lis=sorted(flat_dist_mat)
            rank_n=sorted_lis[:topN]
            for x in rank_n:
                idx=flat_dist_mat.index(x)
                print(gallery[idx+1+q])
            print()



if __name__ == '__main__':
    path_dataset='/home/josemiki/FF-PRID-2020/reid/video1'
    reid = personReIdentifier()
    reid.PersonReIdentification(path_dataset,4)
    