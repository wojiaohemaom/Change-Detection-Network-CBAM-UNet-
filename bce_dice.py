# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:00:17 2022

@author: 何毛毛
"""


from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    """
    加权后的dice coefficient
    """
    y_true = y_true[:, :, :, -
                    1]  # y_true[:, :, :, :-1]=y_true[:, :, :, -1] if dim(3)=1 等效于[8,256,256,1]==>[8,256,256]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    # not working better using mean
    return ((2. * intersection + smooth) / (union + smooth))


def dice_coef_loss(y_true, y_pred):
    """
    目标函数
    """
    return 1 - dice_coef(y_true, y_pred)


def weighted_bce_dice_loss(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(
        y_true, y_pred), axis=[0, 1, 2])

    # note that the weights can be computed automatically using the training smaples
    class_weights = [0.1, 0.9]
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return weighted_bce + 0.5 * (dice_coef_loss(y_true, y_pred))
