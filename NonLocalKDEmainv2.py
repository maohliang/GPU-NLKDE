# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 08:06:38 2022

@author: YanLi
"""
import os
import csv
import numpy as np
import NonLocalKDE_modelv3
import kdeMartix_modelv2
import time
latMax = 39.22
latMin = 37.16
lonMax = 123.17
lonMin = 117.6
# AIS数据集中存储
aisFile = 'C:/Users/Administrator/Desktop/GPU-Based-ADP-NLKDE/AISData/'
fName = ['chengshanjiao/WGS84/','tianjingang/WGS84/']
date = ['2018-01/','2018-07/','2018-10/']
dataAISLon = []
dataAISLat = []
# (1)用于识别每段轨迹在向量dataAIS中的位置
posL = [0]
#
posData = 0
for b in fName:
    fileStr = aisFile + b
    for a in date:
        subFile = os.listdir(fileStr + a)
        for i in range(len(subFile)):
            csvFile = fileStr + a + subFile[i] + '/'
            subCsvFile = os.listdir(csvFile)
            for j in range(len(subCsvFile)):
                fname = csvFile + subCsvFile[j]
                csvData = csv.reader(open(fname,'r'))
                sumData = 0
                for data in csvData:
                    latM = float(data[2])
                    lonM = float(data[3])
                    dataAISLon.append(lonM)
                    dataAISLat.append(latM)
                    sumData += 1
                posL.append(posL[posData] + sumData)
                posData += 1
#
ifInter = 'Y' # 该参数表示计算kde密度矩阵时是否需要插值:'Y':需要;'N':不需要
# kde矩阵的尺寸
sizeX = 500
sizeY = 500
#
ds = 3 # 领域矩阵块的大小,也是高斯卷积核的尺寸,矩阵的实际大小为(ds * 2 + 1) * (ds * 2 + 1)
Ds = 8 # 搜索矩阵块的大小,矩阵的实际大小为(Ds * 2 + 1) * (Ds * 2 + 1)
sigma = 1.5 # 构建高斯卷积核的参数
h = 10 # h越大去噪效果越好,但是图像越模糊,反之h越小去噪效果越差,但去噪之后的失真度越小.
# 运算时间统计
timeL = np.zeros((1,30),dtype = np.float32)
for i in range(31):
    time_start = time.time()
    # 计算kde密度矩阵
    kdeMartix = kdeMartix_modelv2.cau_kdeMartix(dataAISLon, dataAISLat, posL, latMax, latMin, lonMax, lonMin, sizeX, sizeY, ifInter)
    # nonlocal卷积运算
    nonLocalKde = NonLocalKDE_modelv3.cau_nonLocalKde(kdeMartix, sigma, ds, Ds, h)
    time_end = time.time()
    time_sum = time_end - time_start
    if i == 0:
        print("CPU Time: {}".format(time_sum))
    else:
        timeL[0][i-1] = time_sum
        print("CPU Time: {}".format(time_sum))