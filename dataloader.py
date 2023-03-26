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
        RandomPoint = 1024*4
        pointCut = 1024
        point_set_whole = np.loadtxt(self.rootdirs[index])  # .astype(np.float32)#序列
        point_set_whole = point_set_whole[:, :4]  # 前四列
        index2 = np.arange(0, len(point_set_whole), 0).reshape(len(point_set_whole), 1)
        point_set_whole = np.asarray(point_set_whole).astype(np.float32)  # 完整
        result = np.concatenate((point_set_whole, index2), axis=1)  # 加序列

        point_set_cut = result[result[:, 3] == 0][:, :]  # 筛选缺失部分
        cut_rdc = np.random.choice(len(point_set_cut), pointCut, replace=False)  # 缺失部分剪裁成1024个点
        point_set_cut = point_set_cut[cut_rdc, :]
        point_set_cut_one = np.copy(point_set_cut)
        point_set_cut_one[:, [0, 1, 2]] = 1

        rest_set_cut = result[result[:, 3] == 1][:, :]  # 筛选剩余部分
        whole_rdc = np.random.choice(len(rest_set_cut), RandomPoint - pointCut,
                                     replace=False)  # 随机RandomPoint-pointCut个点，随机10240-1024个点
        rest_set_cut = rest_set_cut[whole_rdc, :]  # 剪裁RandomPoint个点

        whole = np.concatenate((rest_set_cut, point_set_cut), axis=0)
        whole = whole[np.argsort(whole[:, 4], ), :]  # 按最后一列排序
        rest = np.concatenate((rest_set_cut, point_set_cut_one), axis=0)
        rest = rest[np.argsort(rest[:, 4], ), :]  # 按最后一列排序

        np.savetxt('output_whole.txt', whole, fmt="%.4f")
        np.savetxt('output_rest.txt', rest, fmt="%.4f")
        np.savetxt('output_cut.txt', point_set_cut, fmt="%.4f")

        complete = torch.from_numpy(whole[:,:3])
        complete = complete.float()
        rest = torch.from_numpy(rest[:,:3])
        rest = rest.float()
        point_set_cut = torch.from_numpy(point_set_cut[:,:3])
        point_set_cut = point_set_cut.float()

        return complete, point_set_cut, point_set_cut, rest

    def __len__(self):
        return len(self.rootdirs)


if __name__ == '__main__':
    dset = PartDataset(root='C:/Users/hs/Desktop/pointcloud/dataset/', classification=True, class_choice=None,
                       npoints=4096, split='train')
    # d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    for i, item in enumerate(dset):
        print(i, item)
