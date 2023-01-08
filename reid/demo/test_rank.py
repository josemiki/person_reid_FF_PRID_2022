import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer



cudnn.benchmark = True
#setup_logger(name="fastreid")

#logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(config_file,opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    #print("args.opts: ",args.opts)
    cfg.freeze()
    return cfg

def setup_cfg2(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

#python demo/test_rank.py --config-file logs/market1501/agw_R50/config.yaml 
# --parallel --vis-label --dataset-name 'MyMarket' 
# --output logs/agw_mymarket_vis 
# --opts MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default="logs/dukemtmc/agw_R50/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        default='MyMarket',
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="descending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default={'MODEL.WEIGHTS', 'logs/market1501/agw_R50/model_final.pth'},
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    #args = get_parser().parse_args()
    #cfg = setup_cfg('logs/market1501/agw_R50/config.yaml',['MODEL.WEIGHTS', 'logs/market1501/agw_R50/model_final.pth'])
    cfg = setup_cfg('logs/dukemtmc/agw_R50/config.yaml',['MODEL.WEIGHTS', 'logs/dukemtmc/agw_R50/model_final.pth'])
    #test_loader, num_query ,test_items= build_reid_test_loader(cfg, args.dataset_name)
    test_loader, num_query ,test_items= build_reid_test_loader(cfg, 'MyMarket')
    
    #demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    demo = FeatureExtractionDemo(cfg, parallel=True)

    #logger.info("Start extracting image features")
    print("Start extracting image features")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()
    
    
    print("len(distmat): ",len(distmat.flatten()))
    flat_dist_mat=distmat.flatten().tolist()
    sorted_lis=sorted(flat_dist_mat)
    print(sorted_lis)
    rank_5=sorted_lis[:5]
    for x in rank_5:
        idx=flat_dist_mat.index(x)
        print(test_items[idx+1])
        

'''
import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader2
from demo.predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer



cudnn.benchmark = True
setup_logger(name="fastreid")
logger = logging.getLogger('fastreid.visualize_result')
cfg_path='logs/dukemtmc/agw_R50/config.yaml'
weights_path='logs/dukemtmc/agw_R50/model_final.pth'
path_dataset='/home/josemiki/FF-PRID-2020/fast-reid/video1'
#path_dataset='/home/josemiki/video1'
dataset_name='MyMarket'
topN=5
#cfg_path2='logs/market1501/agw_R50/config.yaml'
#weights_path2='logs/market1501/agw_R50/model_final.pth'


def setup_cfg(config_file,opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    #print("args.opts: ",args.opts)
    cfg.freeze()
    return cfg

#python demo/test_rank.py --config-file logs/market1501/agw_R50/config.yaml 
# --parallel --vis-label --dataset-name 'MyMarket' 
# --output logs/agw_mymarket_vis 
# --opts MODEL.WEIGHTS logs/market1501/agw_R50/model_final.pth

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default="logs/dukemtmc/agw_R50/config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        default='MyMarket',
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="descending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default={'MODEL.WEIGHTS', 'logs/market1501/agw_R50/model_final.pth'},
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    #args = get_parser().parse_args()
    #cfg = setup_cfg('logs/market1501/agw_R50/config.yaml',['MODEL.WEIGHTS', 'logs/market1501/agw_R50/model_final.pth'])
    cfg = setup_cfg(cfg_path,['MODEL.WEIGHTS', weights_path])
    #test_loader, num_query ,test_items= build_reid_test_loader(cfg, args.dataset_name)
    test_loader, num_query ,gallery= build_reid_test_loader2(cfg, dataset_name, path_dataset)
    
    #demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    demo = FeatureExtractionDemo(cfg, parallel=True)

    #logger.info("Start extracting image features")
    print("Start extracting image features")
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
    #q_pids = np.asarray(pids[:num_query])
    #g_pids = np.asarray(pids[num_query:])
    #q_camids = np.asarray(camids[:num_query])
    #g_camids = np.asarray(camids[num_query:])

    # compute cosine distance
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()
    
    print("gallery: ",len(gallery))
    for q in range(len(q_feat)):
        #print("len(q_feat): ",len(q_feat))
        flat_dist_mat=distmat[q].flatten().tolist()
        sorted_lis=sorted(flat_dist_mat)
        #print(sorted_lis)
        rank_n=sorted_lis[:topN]
        for x in rank_n:
            idx=flat_dist_mat.index(x)
            print(gallery[idx+1+q])
        print()
            
    
    
'''