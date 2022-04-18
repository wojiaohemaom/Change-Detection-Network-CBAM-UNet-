# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:18:57 2022

@author: 何毛毛
"""


from keras_preprocessing import image
import numpy as np


def train_process(file_path, not_label=True):  # 将图像读取为数组
    bs = []  # 为将每张图片合并为一个数组
    if not_label:
        for i in range(len(file_path)):
            data = image.load_img(file_path[i], color_mode='rgb')  # 返回 ndarray
            img_array = image.img_to_array(data)
            bs.append(img_array[np.newaxis])
        x = np.concatenate(bs, axis=0)  # 按照0轴连接
    else:
        for i in range(len(file_path)):
            data = image.load_img(
                file_path[i], color_mode='grayscale')  # 返回 ndarray
            data = data.convert('1')
            img_array = image.img_to_array(data)
            bs.append(img_array[np.newaxis])
        x = np.concatenate(bs, axis=0)  # 按照0轴连接
    return x