# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:11:35 2022

@author: 何毛毛
"""
from keras import layers
from keras.layers import Add, Activation, Conv2D, BatchNormalization
from keras.layers import GlobalMaxPool2D, GlobalAveragePooling2D, Dense, Reshape, Concatenate
from keras.regularizers import l2
import tensorflow as tf

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    # x = Dropout(0.2, name='dp' + stage + '_1')(x)
    x0 = x
    # much better than dropout
    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    # x = Dropout(0.2, name='dp' + stage + '_2')(x)
    x = BatchNormalization(name='bn' + stage + '_2')(x)

    # Channel Attention
    avgpool = GlobalAveragePooling2D(name=stage+'_channel_avgpool')(x)
    maxpool = GlobalMaxPool2D(name=stage+'_channel_maxpool')(x)
    # Shared MLP
    Dense_layer1 = Dense(nb_filter//8, activation='relu',
                         name=stage+'_channel_fc1')
    Dense_layer2 = Dense(nb_filter, activation='relu',
                         name=stage+'_channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = layers.add([avg_out, max_out])
    channel = Activation('sigmoid', name=stage+'_channel_sigmoid')(channel)
    channel = Reshape((1, 1, nb_filter), name=stage +
                      '_channel_reshape')(channel)
    channel_out = layers.Multiply()([x, channel])

    # Spatial Attention
    #avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name=stage+'_spatial_avgpool')
    avgpool = layers.Lambda(tf.reduce_mean, arguments={
                            'axis': 3, 'keepdims': True, 'name': stage+'_spatial_avgpool'})(channel_out)
    #maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name=stage+'_spatial_maxpool')
    maxpool = layers.Lambda(tf.reduce_max, arguments={
                            'axis': 3, 'keepdims': True, 'name': stage+'_spatial_avgpool'})(channel_out)
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (7, 7), strides=1, padding='same',
                     name=stage+'_spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name=stage+'_spatial_sigmoid')(spatial)

    CBAM_out = layers.Multiply()([channel_out, spatial_out])
    if mode == 'residual':
        # x=Add(name='resi'+stage)([x,input_tensor])# 维度不相同！
        x = Add(name='resi' + stage)([CBAM_out, x0])
        return x
    return CBAM_out 