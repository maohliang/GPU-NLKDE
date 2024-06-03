# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 01:29:51 2022

@author: YanLi
"""

import numpy as np
import math
import copy
from pycuda import gpuarray
from pycuda.compiler import SourceModule
# 矩阵的行互换,相当于matlab中函数flipud的功能
def swapRows(kdeMartix):
    m = kdeMartix.shape[0] # 矩阵的行数
    for i in range(math.floor(m/2)):
        transforData = kdeMartix[i]
        transforDataCopy = copy.deepcopy(transforData)
        kdeMartix[i] = kdeMartix[m-i-1]
        kdeMartix[m-i-1] = transforDataCopy
# 计算kde密度矩阵
def cau_kdeMartix(dataAISLon, dataAISLat, posL, latMax, latMin, lonMax, lonMin, sizeX, sizeY):
    # 经纬度向量数据类型转换
    dataAISLon = np.array(dataAISLon)
    dataAISLat = np.array(dataAISLat)
    dataAISLon = np.float32(dataAISLon)
    dataAISLat = np.float32(dataAISLat)
    # 将数据dataAIS从内存传入显存
    dataAISLonGPU = gpuarray.to_gpu(dataAISLon)
    dataAISLatGPU = gpuarray.to_gpu(dataAISLat)
    # outDataGPUF = outDataGPUF + outDataGPU
    # 每个轨迹计算完KDE密度矩阵后,将结果叠加到outDataGPUF矩阵中
    outDataGPUF = gpuarray.zeros((sizeY,sizeX),dtype=np.int32)
    # 内核函数
    # outData[30][30] = outData[sizeY][sizeX]
    ker = SourceModule("""
                        __global__ void conway_ker(int outData[800][800], int *posLon, int *posLat, float *LonL, \
                                                   float *LatL, float lonMax, float lonMin, float latMax, \
                                                       float latMin, float sizeX, float sizeY)
                        {
                            #include <math.h>
                            int thd = blockIdx.x * blockDim.x + threadIdx.x;
                            float xData;
                            float yData;
                            int x;
                            int y;
                            float lonDis = (lonMax - lonMin) / sizeX;
                            float latDis = (latMax - latMin) / sizeY;
                            float lonData = LonL[thd];
                            float latData = LatL[thd];
                            xData = (lonData - lonMin) / lonDis;
                            x = ceil(xData) - 1;
                            yData = (latData - latMin) / latDis;
                            y = ceil(yData) - 1;
                            outData[y][x] = 1;
                            posLon[thd] = x;
                            posLat[thd] = y;
                        }
                        """)
    inp = SourceModule("""
                       __device__ float cauInpY(int x1, int y1, int x2, int y2, int x)
                       {
                           float xNew1 = x1;
                           float yNew1 = y1;
                           float xNew2 = x2;
                           float yNew2 = y2;
                           float xNew = x;
                           float k = (yNew2 - yNew1) / (xNew2 - xNew1);
                           float b = yNew1 - k * xNew1;
                           float y = k * xNew + b;
                           return y;
                           
                        }
                       __global__ void conway_inp(int outData[800][800], int *posLon, int *posLat, int lengthData)
                       {
                           #include <math.h>
                           // 参数lengthData表示向量posLonY和向量posLatX的长度
                           int thd = blockIdx.x * blockDim.x + threadIdx.x;
                           if (thd < lengthData - 1)
                           {
                               int pointY1 = posLat[thd];
                               int pointX1 = posLon[thd];
                               int pointY2 = posLat[thd+1];
                               int pointX2 = posLon[thd+1];
                               if (pointY1 != pointY2 && pointX1 != pointX2)
                               {
                                   if (pointX2 - pointX1 > 1)
                                   {
                                       for (int xCount = pointX1+1; xCount < pointX2; xCount++)
                                       {
                                           float yCount = cauInpY(pointX1,pointY1,pointX2,pointY2,xCount);
                                           int yData = ceil(yCount);
                                           outData[yData][xCount] = outData[yData][xCount] + 1;
                                        }
                                    }
                                   else if (pointX2 - pointX1 < -1)
                                   {
                                       for (int xCount = pointX2+1; xCount < pointX1; xCount++)
                                       {
                                           float yCount = cauInpY(pointX1,pointY1,pointX2,pointY2,xCount);
                                           int yData = ceil(yCount);
                                           outData[yData][xCount] = outData[yData][xCount] + 1;
                                        }
                                    }
                                }
                            }
                        }
                       """)
    conway_ker = ker.get_function("conway_ker")
    conway_inp = inp.get_function("conway_inp")
    posLonAll = gpuarray.zeros(posL[-1],dtype=np.int32)
    posLatAll = gpuarray.zeros(posL[-1],dtype=np.int32)
    for i in range(len(posL) - 1):
        # 存储每个经纬度坐标点对应到kde密度矩阵的x和y的位置坐标
        outDataGPU = gpuarray.zeros((sizeY,sizeX),dtype=np.int32)
        posLon = posLonAll[posL[i]:posL[i+1]]
        posLat = posLatAll[posL[i]:posL[i+1]]
        conway_ker(outDataGPU, posLon, posLat, dataAISLonGPU[posL[i]:posL[i+1]], dataAISLatGPU[posL[i]:posL[i+1]], 
                   np.float32(lonMax), np.float32(lonMin), np.float32(latMax), np.float32(latMin), np.float32(sizeX), \
                       np.float32(sizeY),  grid = (posL[i+1] - posL[i],1,1), block = (1,1,1))
        conway_inp(outDataGPU, posLon, posLat, np.int32(posL[i+1] - posL[i]), \
                   grid = (posL[i+1] - posL[i],1,1), block = (1,1,1))
        outDataGPUF = outDataGPUF + outDataGPU
    # 将最后计算得到kde密度矩阵从显存传到内存
    kdeMartix = outDataGPUF.get()
    swapRows(kdeMartix)
    return kdeMartix