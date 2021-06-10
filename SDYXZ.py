# python
# -*- coding:utf-8 -*-
#@Author: xyx
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import pickle

PI = 3.14

def padding(x, H, W):
    out = np.zeros((2*H, 2*W))
    for i in range(H):
        for j in range(W):
            out[i][j] = x[i][j]
    return out
# 加padding，H, W -> 2H, 2W

def shift(x, P, Q):
    for i in range(P):
        for j in range(Q):
            x[i][j] = x[i][j] * (-1)**(i+j)
    return x
# 使傅里叶变换位于中心

def cut(x, H, W):
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            out[i][j] = x[i][j]
    return out
# 剪切操作

def DFT(img_in, H, W):
    out_real = np.empty((H, W), np.uint8)
    out_imagin = np.empty((H, W))
    for u in range(H):
        for v in range(W):
            real = 0.0
            imagin = 0.0
            for i in range(H):
                for j in range(W):
                    I = img_in[i][j]
                    x = PI * 2 * (i*u/H + j*v/W)
                    real = real + math.cos(x) * I
                    imagin = imagin - math.sin(x) * I
            out_real[u][v] = real
            out_imagin[u][v] = imagin
    return out_real, out_imagin
# 傅里叶变换，使用欧拉公式，n^4
def IDF(in_real, in_imagin, H, W):
    out_real = np.empty((H, W))
    out_imagin = np.empty((H, W))
    for u in range(H):
        for v in range(W):
            real = 0.0
            imagin = 0.0
            for i in range(H):
                for j in range(W):
                    R = in_real[i][j]
                    I = in_imagin[i][j]
                    x = PI * 2 * (i * u / H + j * v / W)
                    real += R * math.cos(x) - I * math.sin(x)
                    imagin += I * math.cos(x) + R * math.sin(x)
            out_real[u][v] = real/(H*W)
            out_imagin[u][v] = imagin/(H*W)
    return out_real, out_imagin
# 逆傅里叶变换


# 第一步读入图片
img = cv2.imread('in.jpg', 0)
# 第二步：进行数据类型转换
img_float = np.float32(img)
print(img.shape)
(H, W) = img.shape
# 原图太大，由于使用的是对照欧拉公式的傅里叶变换，复杂度为n^4，所以太大会很慢
img_float = cv2.resize(img_float,(60,40))
H = 40
W = 60
img_float = padding(img_float, H, W)
print(img_float)
# 移动到中心
img_float = shift(img_float, 2*H, 2*W)
print('img_float',img_float)
# 傅里叶变换
dft_real, dft_imagin = DFT(img_float, 2*H, 2*W)
with open("DFT_real.pkl", "wb") as f:
    pickle.dump(dft_real, f)
with open("DFT_imagin.pkl", "wb") as f:
    pickle.dump(dft_imagin, f)
print('dft finish')

# 第五步：定义频域核：生成的掩模中间为1周围为0
crow, ccol = int(H), int(W) # 求得图像的中心点位置
mask = np.zeros((H*2, W*2), np.uint8)
mask[crow-10:crow+10, ccol-10:ccol+10] = 1
print(mask)
# 第六步：将核与傅里叶变化后图像相乘，保留中间部分
mask_img_real = dft_real * mask
mask_img_imagin = dft_imagin * mask
print('start idf')
# 对“卷积”后输出做逆傅里叶
img_out_real, img_out_imagin = IDF(mask_img_real, mask_img_imagin, 2*H, 2*W)
img_out = shift(img_out_real, 2*H, 2*W)
with open("IDFT_real.pkl", "wb") as f:
    pickle.dump(img_out_real, f)
with open("IDFT_imagin.pkl", "wb") as f:
    pickle.dump(img_out_imagin, f)
# 只取出实部，剪切
img_out = cut(img_out, H, W)
# with open('DFT_real.pkl', 'rb') as fr:
#     dft_real = pickle.load(fr)
# with open('IDFT_real.pkl', 'rb') as fr:
#     img_out_real = pickle.load(fr)
# with open('IDFT_imagin.pkl', 'rb') as fr:
#     img_out_imagin = pickle.load(fr)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(img_out, cmap='gray')
plt.show()
