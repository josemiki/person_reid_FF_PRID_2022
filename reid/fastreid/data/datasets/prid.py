# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

#from reid.fastreid.data.datasets import DATASET_REGISTRY
#from reid.fastreid.data.datasets.bases import ImageDataset

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

__all__ = ['PRID', ]


@DATASET_REGISTRY.register()
class PRID(ImageDataset):
    """PRID
    """
    dataset_dir = "prid_2011"
    dataset_name = 'prid'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        
        self.train_path = os.path.join(self.root, self.dataset_dir, 'slim_train')
        
        self.shoot='single_shot' #multi_shot

        self.query_dir = osp.join(self.root, self.dataset_dir,self.shoot, 'cam_a')
        self.gallery_dir = osp.join(self.root, self.dataset_dir,self.shoot, 'cam_b')

        required_files = [#self.train_path,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        #train = self.process_train(self.train_path)
        query = self.process_dir(self.query_dir, 'query', is_train=False)
        gallery = self.process_dir(self.gallery_dir, 'gallery', is_train=False)

        super().__init__([], query, gallery, **kwargs)
        #super().__init__(train, [], [], **kwargs)

    def process_train(self, train_path):
        data = []
        for root, dirs, files in os.walk(train_path):
            for img_name in filter(lambda x: x.endswith('.png'), files):
                img_path = os.path.join(root, img_name)
                pid = self.dataset_name + '_' + root.split('/')[-1].split('_')[1]
                camid = self.dataset_name + '_' + img_name.split('_')[0]
                data.append([img_path, pid, camid])
        return data

    def process_dir(self, dir_path, shoot, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            img_path2=img_path.replace(".png","_1.png")
            pid, camid = map(int, pattern.search(img_path2).groups())
            #if pid == -1:
            #    continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            if shoot=='query':
                camid -= 1
            else:
                continue #camid -= 1  # index starts from 0
            
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, pid, camid))

        return data