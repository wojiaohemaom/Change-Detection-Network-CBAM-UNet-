# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:14:40 2022

@author: 何毛毛
"""

from keras import Input, Model
from keras.layers import Activation, UpSampling2D, Conv2D, MaxPooling2D, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from res_cbam import standard_unit
from bce_dice import weighted_bce_dice_loss
def unet3(input_shape, deep_supervision=False):
    nb_filter = [64, 128, 256, 512, 1024]
    UpChannels = 320
    mode = 'residual'  # mode='residual' seems to improve better than DS
    # nb_filter = [16, 32, 64, 128, 256]
    # Handle Dimension Ordering for different backends
    bn_axis = 3
    inputs = Input(shape=input_shape)
## -------------Encoder--------------
    h1 = standard_unit(inputs, stage='h1',
                       nb_filter=nb_filter[0], mode=mode)  # 256*256*64
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(h1)

    h2 = standard_unit(pool1, stage='h2', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D((2, 2), strides=(
        2, 2), name='pool2')(h2)  # (128*128*128)

    h3 = standard_unit(pool2, stage='h3', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D((2, 2), strides=(
        2, 2), name='pool3')(h3)  # (64*64*256)

    h4 = standard_unit(pool3, stage='h4', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D((2, 2), strides=(
        2, 2), name='pool4')(h4)  # (32*32*512)

    h5 = standard_unit(pool4, stage='h5',
                       nb_filter=nb_filter[4], mode=mode)   # (16*16*1025)
    ## -------------Decoder-------------
    '''stage 4d'''
    # h1->256*256, hd4->32*32, Pooling 8 times
    h1_h4 = MaxPooling2D((8, 8), strides=(8, 8), name='h1_h4')(h1)
    con_h1_h4 = Conv2D(nb_filter[0], (3, 3), name='con_h1_h4',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h1_h4)
    bn_h1_h4 = BatchNormalization(name='bn_h1_h4')(con_h1_h4)
    act_h1_h4 = Activation('relu')(bn_h1_h4)  # 32*32*64

    # h2->128*128, hd4->32*32, Pooling 4 times
    h2_h4 = MaxPooling2D((4, 4), strides=(4, 4), name='h2_h4')(h2)
    con_h2_h4 = Conv2D(nb_filter[0], (3, 3), name='con_h2_h4',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h2_h4)
    bn_h2_h4 = BatchNormalization(name='bn_h2_h4')(con_h2_h4)
    act_h2_h4 = Activation('relu')(bn_h2_h4)  # 32*32*64

    # h3->64*64, hd4->32*32, Pooling 2 times
    h3_h4 = MaxPooling2D((2, 2), strides=(2, 2), name='h3_h4')(h3)
    con_h3_h4 = Conv2D(nb_filter[0], (3, 3), name='con_h3_h4',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h3_h4)
    bn_h3_h4 = BatchNormalization(name='bn_h3_h4')(con_h3_h4)
    act_h3_h4 = Activation('relu')(bn_h3_h4)  # 32*32*64

    # h4->32*32, hd4->32*32, Concatenation
    con_h4_h4 = Conv2D(nb_filter[0], (3, 3), name='con_h4_h4',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h4)
    bn_h4_h4 = BatchNormalization(name='bn_h4_h4')(con_h4_h4)
    act_h4_h4 = Activation('relu')(bn_h4_h4)  # 32*32*64

    # hd5->16*16, hd4->32*32, Upsample 2 times
    h5_h4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(h5)
    con_h5_h4 = Conv2D(nb_filter[0], (3, 3), name='con_h5_h4',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h5_h4)
    bn_h5_h4 = BatchNormalization(name='bn_h5_h4')(con_h5_h4)
    act_h5_h4 = Activation('relu')(bn_h5_h4)  # 32*32*64

    pic4 = concatenate([act_h1_h4, act_h2_h4, act_h3_h4, act_h4_h4,
                        act_h5_h4], name='pic4', axis=bn_axis)  # 32*32*320
    con_pic4 = Conv2D(UpChannels, (3, 3), name='con_pic4',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(pic4)
    bn_pic4 = BatchNormalization(name='bn_pic4')(con_pic4)
    act_pic4 = Activation('relu')(bn_pic4)
    '''stage 3d'''
    # h1->256*256, hd3->64*64, Pooling 4 times
    h1_h3 = MaxPooling2D((4, 4), strides=(4, 4), name='h1_h3')(h1)
    con_h1_h3 = Conv2D(nb_filter[0], (3, 3), name='con_h1_h3',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h1_h3)
    bn_h1_h3 = BatchNormalization(name='bn_h1_h3')(con_h1_h3)
    act_h1_h3 = Activation('relu')(bn_h1_h3)  # 64*64*64

    # h2->128*128, hd3->64*64, Pooling 2 times
    h2_h3 = MaxPooling2D((2, 2), strides=(2, 2), name='h2_h3')(h2)
    con_h2_h3 = Conv2D(nb_filter[0], (3, 3), name='con_h2_h3',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h2_h3)
    bn_h2_h3 = BatchNormalization(name='bn_h2_h3')(con_h2_h3)
    act_h2_h3 = Activation('relu')(bn_h2_h3)  # 64*64*64

    # h3->64*64, hd3->64*64, Concatenation
    con_h3_h3 = Conv2D(nb_filter[0], (3, 3), name='con_h3_h3',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h3)
    bn_h3_h3 = BatchNormalization(name='bn_h3_h3')(con_h3_h3)
    act_h3_h3 = Activation('relu')(bn_h3_h3)  # 64*64*64

    # h4->32*32, hd3->64*64, Upsample 2 times
    h4_h3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(act_pic4)
    con_h4_h3 = Conv2D(nb_filter[0], (3, 3), name='con_h4_h3',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h4_h3)
    bn_h4_h3 = BatchNormalization(name='bn_h4_h3')(con_h4_h3)
    act_h4_h3 = Activation('relu')(bn_h4_h3)  # 64*64*64

    # h5->16*16, hd3->64*64, Upsample 4 times
    h5_h3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(h5)
    con_h5_h3 = Conv2D(nb_filter[0], (3, 3), name='con_h5_h3',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h5_h3)
    bn_h5_h3 = BatchNormalization(name='bn_h5_h3')(con_h5_h3)
    act_h5_h3 = Activation('relu')(bn_h5_h3)  # 64*64*64

    pic3 = concatenate([act_h1_h3, act_h2_h3, act_h3_h3, act_h4_h3,
                        act_h5_h3], name='pic3', axis=bn_axis)  # 64*64*320
    con_pic3 = Conv2D(UpChannels, (3, 3), name='con_pic3',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(pic3)
    bn_pic3 = BatchNormalization(name='bn_pic3')(con_pic3)
    act_pic3 = Activation('relu')(bn_pic3)
    '''stage 2d'''
    # h1->256*256, hd2->128*128, Pooling 2 times
    h1_h2 = MaxPooling2D((2, 2), strides=(2, 2), name='h1_h2')(h1)
    con_h1_h2 = Conv2D(nb_filter[0], (3, 3), name='con_h1_h2',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h1_h2)
    bn_h1_h2 = BatchNormalization(name='bn_h1_h2')(con_h1_h2)
    act_h1_h2 = Activation('relu')(bn_h1_h2)  # 128*128*64

    # h2->128*128, hd2->128*128,  Concatenation
    con_h2_h2 = Conv2D(nb_filter[0], (3, 3), name='con_h2_h2',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h2)
    bn_h2_h2 = BatchNormalization(name='bn_h2_h2')(con_h2_h2)
    act_h2_h2 = Activation('relu')(bn_h2_h2)  # 128*128*64

    # h3->64*64, hd2->128*128, Upsample 2 times
    h3_h2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(act_pic3)
    con_h3_h2 = Conv2D(nb_filter[0], (3, 3), name='con_h3_h2',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h3_h2)
    bn_h3_h2 = BatchNormalization(name='bn_h3_h2')(con_h3_h2)
    act_h3_h2 = Activation('relu')(bn_h3_h2)  # 128*128*64

    # h4->32*32, hd2->128*128, Upsample 4 times
    h4_h2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(act_pic4)
    con_h4_h2 = Conv2D(nb_filter[0], (3, 3), name='con_h4_h2',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h4_h2)
    bn_h4_h2 = BatchNormalization(name='bn_h4_h2')(con_h4_h2)
    act_h4_h2 = Activation('relu')(bn_h4_h2)  # 128*128*64

    # h5->16*16,  hd2->128*128, Upsample 8 times
    h5_h2 = UpSampling2D(size=(8, 8), interpolation='bilinear')(h5)
    con_h5_h2 = Conv2D(nb_filter[0], (3, 3), name='con_h5_h2',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h5_h2)
    bn_h5_h2 = BatchNormalization(name='bn_h5_h2')(con_h5_h2)
    act_h5_h2 = Activation('relu')(bn_h5_h2)  # 128*128*64

    pic2 = concatenate([act_h1_h2, act_h2_h2, act_h3_h2, act_h4_h2,
                        act_h5_h2], name='pic2', axis=bn_axis)  # 128*128*320
    con_pic2 = Conv2D(UpChannels, (3, 3), name='con_pic2',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(pic2)
    bn_pic2 = BatchNormalization(name='bn_pic2')(con_pic2)
    act_pic2 = Activation('relu')(bn_pic2)
    '''stage 1d'''
    # h1->256*256, h1->256*256, Concatenation
    con_h1_h1 = Conv2D(nb_filter[0], (3, 3), name='con_h1_h1',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h1)
    bn_h1_h1 = BatchNormalization(name='bn_h1_h1')(con_h1_h1)
    act_h1_h1 = Activation('relu')(bn_h1_h1)  # 64*64*64

    # h2->128*128, h1->256*256,  Upsample 2 times
    h2_h1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(act_pic2)
    con_h2_h1 = Conv2D(nb_filter[0], (3, 3), name='con_h2_h1',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h2_h1)
    bn_h2_h1 = BatchNormalization(name='bn_h2_h1')(con_h2_h1)
    act_h2_h1 = Activation('relu')(bn_h2_h1)  # 64*64*64

    # h3->64*64, h1->256*256, Upsample 4 times
    h3_h1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(act_pic3)
    con_h3_h1 = Conv2D(nb_filter[0], (3, 3), name='con_h3_h1',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h3_h1)
    bn_h3_h1 = BatchNormalization(name='bn_h3_h1')(con_h3_h1)
    act_h3_h1 = Activation('relu')(bn_h3_h1)  # 64*64*64

    # h4->32*32, h1->256*256, Upsample 8 times
    h4_h1 = UpSampling2D(size=(8, 8), interpolation='bilinear')(act_pic4)
    con_h4_h1 = Conv2D(nb_filter[0], (3, 3), name='con_h4_h1',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h4_h1)
    bn_h4_h1 = BatchNormalization(name='bn_h4_h1')(con_h4_h1)
    act_h4_h1 = Activation('relu')(bn_h4_h1)  # 64*64*64

    # h5->16*16,  h1->256*256, Upsample 16 times
    h5_h1 = UpSampling2D(size=(16, 16), interpolation='bilinear')(h5)
    con_h5_h1 = Conv2D(nb_filter[0], (3, 3), name='con_h5_h1',
                       kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h5_h1)
    bn_h5_h1 = BatchNormalization(name='bn_h5_h1')(con_h5_h1)
    act_h5_h1 = Activation('relu')(bn_h5_h1)  # 64*64*64

    pic1 = concatenate([act_h1_h1, act_h2_h1, act_h3_h1, act_h4_h1,
                        act_h5_h1], name='pic1', axis=bn_axis)  # 256*256*320
    con_pic1 = Conv2D(UpChannels, (3, 3), name='con_pic1',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(pic1)
    bn_pic1 = BatchNormalization(name='bn_pic1')(con_pic1)
    act_pic1 = Activation('relu')(bn_pic1)
    # DeepSup
    out_pic1 = Conv2D(1, (3, 3), name='out_pic1',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(act_pic1)
    out1 = Activation('sigmoid')(out_pic1)
    out_pic2 = Conv2D(1, (3, 3), name='out_pic2',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(act_pic2)
    up_out_pic2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(out_pic2)
    out2 = Activation('sigmoid')(up_out_pic2)
    out_pic3 = Conv2D(1, (3, 3), name='out_pic3',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(act_pic3)
    up_out_pic3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(out_pic3)
    out3 = Activation('sigmoid')(up_out_pic3)
    out_pic4 = Conv2D(1, (3, 3), name='out_pic4',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(act_pic4)
    up_out_pic4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(out_pic4)
    out4 = Activation('sigmoid')(up_out_pic4)
    out_pic5 = Conv2D(1, (3, 3), name='out_pic5',
                      kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(h5)
    up_out_pic5 = UpSampling2D(
        size=(16, 16), interpolation='bilinear')(out_pic5)
    out5 = Activation('sigmoid')(up_out_pic5)

# msof
    conv_fuse = concatenate([out_pic1, up_out_pic2, up_out_pic3,up_out_pic4, up_out_pic5], name='merge_fuse', axis=bn_axis)
    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='output_6',
                  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv_fuse)

    if deep_supervision:
        model = Model(input=inputs, output=[
                      out1, out2, out3, out4, out5, out6])
        model.compile(optimizer=Adam(lr=1e-4),
                      #loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'],
                      loss=[weighted_bce_dice_loss, weighted_bce_dice_loss, weighted_bce_dice_loss, weighted_bce_dice_loss,
                            weighted_bce_dice_loss, weighted_bce_dice_loss],
                      metrics=['binary_accuracy']
                      )
    else:
        model = Model(input=inputs, output=[out1])
        model.compile(optimizer=Adam(lr=1e-4), loss=[weighted_bce_dice_loss],
                      metrics=['binary_accuracy'])
    model.summary()
    return model