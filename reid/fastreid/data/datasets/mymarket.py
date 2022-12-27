# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from pathlib import Path


@DATASET_REGISTRY.register()
class MyMarket(ImageDataset):
    """FF-PRID.

    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_name = "mymarket"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        #print("MyMarket===========0root: ",root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MyMarket')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'train')
        #self.query_dir = osp.join(self.data_dir, 'query')

        self.query_dir = Path(root).parents[1]
        #print("=======self.query_dir:",self.query_dir)

        self.gallery_dir = osp.join(self.data_dir, 'cropps')
        #print("=======self.gallery_dir:",self.gallery_dir)

        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            #self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        #if self.market1501_500k:
        #    required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_q(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        #if self.market1501_500k:
        #    gallery += self.process_dir(self.extra_gallery_dir, is_train=False)

        super(MyMarket, self).__init__(train, query, gallery, **kwargs)
        #super(MyMarket, self).__init__(query, gallery, **kwargs)
    '''
    def process_dir2(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
    '''
    def process_q(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        #print("img_paths:",img_paths)
        #pattern = re.compile(r'[0-9]+')
        pattern = re.compile(r'([\d]+)_([\d])')
        data = []
        for img_path in img_paths:
            img_path2=img_path.replace(".png","_1.png")
            #print("img_path2: ",img_path2)
            pid, camid  = map(int, pattern.search(img_path2).groups())
            #print("pid: ",pid)
            #print("camid:  ",camid)
            #print("img_path: ",img_path)
            data.append((img_path, pid, camid))

        return data
    
    def process_dir2(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        #print("img_paths:",img_paths)
        #pattern = re.compile(r'[0-9]+')
        pattern = re.compile(r'([\d]+)_([\d])')
        data = []
        for img_path in img_paths:
            img_path2=img_path.replace(".jpg","_1.jpg")
            #print("img_path2: ",img_path2)
            pid, camid  = map(int, pattern.search(img_path2).groups())
            #print("pid: ",pid)
            #print("camid:  ",camid)
            #print("img_path: ",img_path)
            data.append((img_path, pid, camid))

        return data
    
    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        #print("img_paths:",img_paths)
        #pattern = re.compile(r'[0-9]+')
        pattern = re.compile(r'([\d]+)_([\d])')
        data = []
        for img_path in img_paths:
            #img_path2=img_path.replace(".jpg","_1.jpg")
            #print("img_path2: ",img_path2)
            pid, camid  = map(int, pattern.search(img_path).groups())
            #print("pid: ",pid)
            #print("camid:  ",camid)
            #print("img_path: ",img_path)
            data.append((img_path, pid, camid))

        return data