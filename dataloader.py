import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import sys
import glob


class PartDataset(data.Dataset):
    def __init__(self, root=None, npoints=2500, classification=False, class_choice=None, split='train', normalize=True):
        root = 'C:/Users/hs/Desktop/pointcloud/dataset/'
        self.data = []
        self.root = root
        self.rootdirs = glob.glob(root + '*')
        self.coloboma = []
        self.complete = []
        self.rest = []


    def __getitem__(self, index):

        point_set_whole = np.loadtxt(self.rootdirs[index])  # .astype(np.float32)
        point_set_whole = np.asarray(point_set_whole).astype(np.float32) #完整
        point_set_cut = point_set_whole[point_set_whole[:, 3] == 0][:, :3] #缺失部分
        point_set_whole = np.random.choice(len(point_set_whole), 10240, replace=True)
        point_set_whole = point_set_whole[point_set_whole, :]
        print(point_set_whole.shape)

        rest = point_set_whole[point_set_whole[:, 3] == 1][:, :3]

        point_set_cut = np.random.choice(len(point_set_cut), 1024, replace=True)#缺失部分剪裁成1024个点
        point_set_cut = point_set_cut[point_set_cut, :]
        print(point_set_cut.shape)
        point_set_whole = point_set_whole[:, :3]



        complete = torch.from_numpy(point_set_whole)
        complete = complete.float()
        rest = torch.from_numpy(rest)

        return complete, point_set_cut, point_set_cut, complete

    def __len__(self):
        return len(self.rootdirs)

if __name__ == '__main__':
    dset = PartDataset(root='C:/Users/hs/Desktop/pointcloud/dataset/', classification=True, class_choice=None,
                       npoints=4096, split='train')
       #d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    for i ,item in enumerate(dset):
        print(i,item)

