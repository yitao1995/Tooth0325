# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:59:36 2023

@author: duanjx
"""
import os
import open3d as o3d
import numpy as np
import pandas as pd


# data_path = './test_one/crop_ours_txt.txt'
data_path = 'test_one/0001陈雪柔_crop_ours.csv'

ext = os.path.splitext(data_path)[-1]
if ext == '.txt':
    points_data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
elif ext == '.csv':
    data = pd.read_csv(data_path, encoding='utf-8')  # 读取csv文件
    data_234 = data.iloc[:, :3]  
    points_data = np.array(data_234)  # 转换为numpy方便计算
else:
    raise ValueError("file format error!!")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])
o3d.visualization.draw_geometries([pcd])

