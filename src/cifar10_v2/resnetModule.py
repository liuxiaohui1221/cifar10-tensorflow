import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import math

"""
用于控制模型层数 https://www.cnblogs.com/hanmk/p/13402910.html
"""


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    Arguments:
        inputs (tensor): 输入
        num_filters (int): 卷积核个数
        kernel_size (int): 卷积核大小
        activation (string): 激活层
        batch_normalization (bool): 是否使用批归一化
        conv_first (bool): conv-bn-active(True) or bn-active-conv (False)层堆叠次序

    Returns:
        x (tensor): 输出
    """
    conv = keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet(input_shape, depth, num_classes=10):
    """ResNet

    Arguments:
        input_shape (tensor): 输入尺寸
        depth (int): 网络层数
        num_classes (int): 预测类别数

    Return:
        model (Model): 模型
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2')
    # 超参数
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = keras.layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = keras.layers.Activation('relu')(x)
        num_filters *= 2
    x = keras.layers.AveragePooling2D(pool_size=8)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='he_normal')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

