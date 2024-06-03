# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:44:39 2022

@author: YanLi
"""

import math
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import pycuda.driver as drv

# 构建高斯卷积核
def get_gussian(ds, sigma):
    kernelSize = ds * 2 + 1 # 构建高斯卷积核,大小与矩阵中心像素点领域大小一致,即为(ds * 2 + 1) * (ds * ds + 1)
    kernelConvolution = np.zeros((kernelSize, kernelSize))
    sumData = 0
    centerXY = ds
    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - centerXY
            y = j - centerXY
            kernelConvolution[i][j] = (1 / (2 * math.pi * (sigma**2))) * math.exp(-(x**2 + y**2) / (2 * (sigma**2)))
            sumData = sumData + kernelConvolution[i][j]
    sumData = 1 / sumData
    kernelGaussian = kernelConvolution * sumData;
    return kernelGaussian

# NonLocalKdeGPU算法
# 参数subsectionNum表示将原始kde密度矩阵分成几个分段矩阵
# 参数subsectionInforL表示对扩展后的kde矩阵进行分割标记
def cau_nonLocalKdeGpu(kdeMartix, sigma, ds, Ds, h, subsectionNum, subsectionInforL, sizeRC):
    # 构建的高斯卷积核函数
    kernelGaussian = get_gussian(ds, sigma)
    kernelGaussian = np.float32(kernelGaussian)
    kernelGaussianGPU = gpuarray.to_gpu(kernelGaussian) # 将高斯卷积核传入显存
    #
    kdeMartix = np.float32(kdeMartix) # 更改数据类型,GPU对于float和int均适配32的
    # 计算kde密度矩阵的最大值和最小值,后续为kde密度矩阵的归一化使用
    maxData = np.max(kdeMartix)
    maxData = np.float32(maxData)
    minData = np.min(kdeMartix)
    minData = np.float32(minData)
    #
    m = kdeMartix.shape[0] # kde密度矩阵的行数
    n = kdeMartix.shape[1] # kde密度矩阵的列数
    offset = ds + Ds
    mExtend = offset * 2 + m # 扩展矩阵后的行数
    nExtend = offset * 2 + n # 扩展矩阵后的列数
    kdeMartixExtend = np.zeros((mExtend, nExtend)) # 将kde密度矩阵进行扩展
    kdeMartixExtend = np.float32(kdeMartixExtend) # 更改数据类型
    kdeMartixExtend[offset:(mExtend - offset), offset:(nExtend - offset)] = kdeMartix
    # 将原始kde密度矩阵进行分解后,每个小矩阵的大小
    subsectionM = np.int32(m / math.sqrt(subsectionNum))
    subsectionN = np.int32(n / math.sqrt(subsectionNum))
    outSubNonLocalKde = np.zeros((subsectionM, subsectionN))
    outSubNonLocalKde = np.float32(outSubNonLocalKde)
    # 将参数ds,Ds,h分别转换为int32,int32,float32类型
    ds = np.int32(ds)
    Ds = np.int32(Ds)
    h = np.float32(h)
    ker = SourceModule("""
                       __device__ float cauConData(int x, int y, float inData[130][130], float kernel[11][11], \
                                                   int ds, int Ds, float h)
                       {
                           #include <math.h>
                           int offset = ds + Ds;
                           int i = x + offset; // 中心点坐标y
                           int j = y + offset; // 中心点坐标x
                           float kernelData = 0.0;
                           // 将以(j,i)为中心的,维度大小与核函数一致的矩阵从inData中提取出来
                           float w1[11][11] = {{0.0}}; //如果卷积核尺寸变换,这个位置需要更改
                           int i1 = 0;
                           for (int iw1 = j - ds; iw1 < j + ds + 1; iw1++)
                           {
                               int j1 = 0;
                               for (int jw1 = i - ds; jw1 < i + ds + 1; jw1++)
                               {
                                   w1[i1][j1] = inData[iw1][jw1];
                                   j1 = j1 + 1;
                                }
                               i1 = i1 + 1;
                            }
                           //  将w1的位置信息(x,y)进行存储
                           float w1posx[11][11] = {{0.0}}; // 如果卷积核尺寸变换,这个位置需要更改
                           int iw1x = 0;
                           float w1posxx = j - ds;
                           while (iw1x < 11) // 如果卷积核尺寸变换,这个位置需要更改
                           {
                               for (int jw1x = 0; jw1x < 11; jw1x++) // 如果卷积核尺寸变换,这个位置需要更改
                               {
                                   w1posx[iw1x][jw1x] = w1posxx;
                                }
                               iw1x = iw1x + 1;
                               w1posxx = w1posxx + 1.0;
                            }
                           float w1posy[11][11] = {{0.0}}; // 如果卷积核尺寸变换,这个位置需要更改
                           int jw1y = 0;
                           float w1posyy = i - ds;
                           while (jw1y < 11) // 如果卷积核尺寸变换,这个位置需要更改
                           {
                               for (int iw1y = 0; iw1y < 11; iw1y++) // 如果卷积核尺寸变换,这个位置需要更改
                               {
                                   w1posy[iw1y][jw1y] = w1posyy;
                                }
                               jw1y = jw1y + 1;
                               w1posyy = w1posyy + 1;
                            }
                           //加权因子矩阵和图像-->SearchWindows
                           float weightMartix[21][21] = {{0.0}}; // 如果搜索窗尺寸变换,这个位置需要更改
                           float searchKdeMartix[21][21] = {{0.0}}; // 如果搜索窗尺寸变换,这个位置需要更改
                           int i2 = 0;
                           for (int is = j - Ds; is < j + Ds + 1; is++)
                           {
                               int j2 = 0;
                               for (int js = i - Ds; js < i + Ds + 1; js++)
                               {
                                   searchKdeMartix[i2][j2] = inData[is][js];
                                   j2 = j2 + 1;
                                }
                               i2 = i2 + 1;
                            }
                           //nonlocal具体算法计算
                           for (int r = -Ds; r < Ds + 1; r++)
                           {
                               for (int s = -Ds; s < Ds + 1; s++)
                               {
                                   //跳过当前点
                                   if (r == 0 && s == 0)
                                   {
                                       continue;
                                    }
                                   //
                                   int ii = j + r;
                                   int jj = i + s;
                                   float w2[11][11] = {{0.0}}; // 如果卷积核尺寸变换,这个位置需要更改
                                   int i3 = 0;
                                   for (int iw2 = ii - ds; iw2 < ii + ds + 1; iw2++)
                                   {
                                       int j3 = 0;
                                       for (int jw2 = jj - ds; jw2 < jj + ds + 1; jw2++)
                                       {
                                           w2[i3][j3] = inData[iw2][jw2];
                                           j3 = j3 + 1;
                                        }
                                       i3 = i3 + 1;
                                    }
                                   // 将w2的位置信息(x,y)进行存储
                                   float w2posx[11][11] = {{0.0}}; // 如果卷积核尺寸变换,这个位置需要更改
                                   int iw2x = 0;
                                   float w2posxx = ii - ds;
                                    while (iw2x < 11) // 如果卷积核尺寸变换,这个位置需要更改
                                    {
                                        for (int jw2x = 0; jw2x < 11; jw2x++) // 如果卷积核尺寸变换,这个位置需要更改
                                        {
                                             w2posx[iw2x][jw2x] = w2posxx;
                                        }
                                        iw2x = iw2x + 1;
                                        w2posxx = w2posxx + 1;
                                    }
                                   float w2posy[11][11] = {{0.0}}; // 如果卷积核尺寸变换,这个位置需要更改
                                   int jw2y = 0;
                                   float w2posyy = jj - ds;
                                   while (jw2y < 11) // 如果卷积核尺寸变换,这个位置需要更改
                                   {
                                       for (int iw2y = 0; iw2y < 11; iw2y++) // 如果卷积核尺寸变换,这个位置需要更改
                                       {
                                           w2posy[iw2y][jw2y] = w2posyy;
                                        }
                                       jw2y = jw2y + 1;
                                       w2posyy = w2posyy + 1;
                                    }
                                   //计算加权因子
                                   float distance = 0.0;
                                   for (int iwM = 0; iwM < 11; iwM++) // 如果卷积核尺寸变换,这个位置需要更改
                                   {
                                       for (int jwM = 0; jwM < 11; jwM++) // 如果卷积核尺寸变换,这个位置需要更改
                                       {
                                           distance = distance + (kernel[iwM][jwM] * \
                                                                  (w1[iwM][jwM] - w2[iwM][jwM]) * \
                                                                      (w1[iwM][jwM] - w2[iwM][jwM])) / \
                                               sqrt((w1posx[iwM][jwM] - w2posx[iwM][jwM]) * \
                                                    (w1posx[iwM][jwM] - w2posx[iwM][jwM]) + \
                                                        (w1posy[iwM][jwM] - w2posy[iwM][jwM]) * \
                                                            (w1posy[iwM][jwM] - w2posy[iwM][jwM]));
                                        }
                                    }
                                   //
                                   weightMartix[r+Ds][s+Ds] = exp(-distance / (h*h));
                                   //
                                }
                            }
                           //将weightMartix矩阵最大值赋给矩阵中心,同时计算weightMartix的和
                           float maxData = -999999.0;
                           float value;
                           float sumDataw = 0.0;
                           for (int iMax = 0; iMax < 21; iMax++) // 如果搜索窗尺寸变换,这个位置需要更改
                           {
                               for (int jMax = 0; jMax < 21; jMax++) // 如果搜索窗尺寸变换,这个位置需要更改
                               {
                                   value = weightMartix[iMax][jMax];
                                   sumDataw = sumDataw + value;
                                   if (value > maxData)
                                   {
                                       maxData = value;
                                    }
                                }
                            }
                           weightMartix[Ds][Ds] = maxData;
                           sumDataw = sumDataw + maxData;
                           //计算kernelData值
                           for (int i4 = 0; i4 < 21; i4++) // 如果搜索窗尺寸变换,这个位置需要更改
                           {
                               for (int j4 = 0; j4 < 21; j4++) // 如果搜索窗尺寸变换,这个位置需要更改
                               {
                                   kernelData = kernelData + (weightMartix[i4][j4] / sumDataw) * searchKdeMartix[i4][j4];
                                }
                            }
                           //
                           return kernelData;
                        }
                       __global__ void conway_ker(float outData[100][100], float inData[130][130], float kernel[11][11], \
                                                  int ds, int Ds, float h, float maxData, float minData)
                       {
                           const int X = threadIdx.x + blockIdx.x * blockDim.x;
                           const int Y = threadIdx.y + blockIdx.y * blockDim.y;
                           // 对原始kde密度矩阵进行归一化处理
                           float inDataDeal[130][130] = {{0.0}};
                           for (int i = 0; i < 130; i++)
                           {
                               for (int j = 0; j < 130; j++)
                               {
                                   inDataDeal[i][j] = (inData[i][j] - minData) / (maxData - minData);
                                }
                            }
                           //
                           float kernelData = cauConData(X, Y, inDataDeal, kernel, ds, Ds, h);
                           outData[Y][X] = kernelData;
                        }
                       """)
    conway_ker = ker.get_function("conway_ker")
    inDataGPU = []
    outDataGPU = []
    # 最终计算得到的NonLocalKde分矩阵从显存传送到内存
    outDataFromGPU = []
    # PyCUDA流
    streams = []
    # 构建PyCUDA流
    for i in range(subsectionNum):
        streams.append(drv.Stream())
    # 将数据从内存传到显存
    for i in range(subsectionNum):
        inDataM = kdeMartixExtend[subsectionInforL[i][0]+offset-(ds+Ds):subsectionInforL[i][1]+offset+(ds+Ds+1),\
                                  subsectionInforL[i][2]+offset-(ds+Ds):subsectionInforL[i][3]+offset+(ds+Ds+1)]
        inDataGPU.append(gpuarray.to_gpu_async(inDataM, stream = streams[i]))
        outDataGPU.append(gpuarray.to_gpu_async(outSubNonLocalKde, stream = streams[i]))
    for i in range(subsectionNum):
        conway_ker(outDataGPU[i], inDataGPU[i], kernelGaussianGPU, ds, Ds, h, maxData, minData, \
                   grid = (10,10,1), block = (10,10,1), stream = streams[i])
    # 将GPU计算得到的数据从显存传到内存
    for i in range(subsectionNum):
        outDataFromGPU.append(outDataGPU[i].get_async(stream = streams[i]))
    # 将分布式计算的所有矩阵块进行合并
    # 矩阵横向合并
    subM = []
    numLable = 0
    for i in range(sizeRC):
        mergeM = np.hstack((outDataFromGPU[numLable + (i * sizeRC)], outDataFromGPU[numLable+1 + (i * sizeRC)]))
        for j in range(numLable+2, sizeRC):
            mergeM = np.hstack((mergeM, outDataFromGPU[j + (i * sizeRC)]))
        numLable = 0
        subM.append(mergeM)
    # 矩阵纵向合并
    outNonLocalKdeF = np.vstack((subM[0],subM[1]))
    for i in range(2,sizeRC):
        outNonLocalKdeF = np.vstack((outNonLocalKdeF,subM[i]))
    return outNonLocalKdeF