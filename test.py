import torch
import torch.utils.data as data
import os
import json
import numpy as np
import glob
#
# catfile = r'C:\Users\hs\Desktop\pointcloud\dataset\1-陈楚雪-1-1 水平.txt'
# ls = None
# data = []
#
# with open(catfile, 'r') as f:
#     for d in f:
#         data.append(d)
    # train_ids = set([str(d.split('/')[3]) ])
# print(data[0])
# print(data[1])
# coloboma = [[1, 7, 1, 1],
#             [2, 8, 0, 1],
#             [3, 9, 0, 0],
#             [4, 0, 1, 0]
#             ]
# complete = []
# rest = []
# point_set = np.loadtxt(catfile)  # .astype(np.float32)
# for i in point_set:
#     complete.append(i[:3])
#     if (i[3]) == 0:
#         coloboma.append(i[:3])
#     else:
#         rest.append(i[:3])
# #print(train_ids)
# point_set = np.loadtxt(catfile)  # .astype(np.float32)
# point_set = np.asarray(point_set).astype(np.float32)
# complete = np.random.choice(len(point_set), 45000, replace=True)
# point_set = point_set[complete, :]
# print(point_set.shape)
# print(point_set[:, 3] == 1)
# a = point_set[point_set[:, 3] == 1]
# b = point_set[point_set[:, 3] == 0]
# c = torch.from_numpy(a)
# d = torch.from_numpy(b)
# Layers = [c.T, d.T]
# print(torch.cat(Layers, dim=1).shape)

# complete = np.asarray(coloboma).astype(np.float32)
# print(complete.shape)
# print(complete[complete[:, 3] == 1])
# print(complete[complete[:, 3] == 0])

# for x in np.nditer(complete, order='C'):
#     print(x)#C order，即是行序优先；
# print(complete.shape)
# choice = np.random.choice(len(complete), 45000, replace=True)
# point_set = point_set[choice, :]

# print(len(point_set))
# print(len(rest))
root = 'C:/Users/hs/Desktop/pointcloud/dataset/'
imgdirs = glob.glob(root + '*')
plist=[]
clist=[]
rlist=[]
for i in imgdirs:
    point_set = np.loadtxt(i)  # .astype(np.float32)
    print(i)
    point_set = np.asarray(point_set).astype(np.float32)
    plist.append(point_set)
for i,j in enumerate(plist):
    clist.append(j[j[:, 3] == 0][:, :3])
    rlist.append(j[j[:, 3] == 1][:, :3])
for j in plist:
    print(j)
