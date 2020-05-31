"""
Implementation of SRResNet
Author: Zixuan Zhang
Date: 5/30/2020
Reference: https://arxiv.org/pdf/1609.04802.pdf
"""

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as layers


class SRResNet(tfk.Model):
    def __init__(self, imgChannel=3, blockNum=6, channel=128):
        super(SRResNet, self).__init__()
        # the first conv block
        self.conv1 = layers.Conv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same')
        self.prelu1 = layers.ReLU()
        self.resBlockList = []

        for i in range(blockNum):
            # original paper # channel = 64
            self.resBlockList.append(ResidualBlock(2, filterNum=channel, kernelSize=(3, 3),
                                                   strideSize=(1, 1)))
        # original paper # channel = 64
        self.conv2 = layers.Conv2D(filters=channel, kernel_size=(3, 3), strides=(1, 1), padding='same')

        self.batchNorm1 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=64 * 16, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.prelu2 = layers.ReLU()

        self.conv4 = layers.Conv2D(filters=64 * 4, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.prelu3 = layers.ReLU()

        self.out = layers.Conv2D(imgChannel, kernel_size=(3, 3), strides=(1, 1),
                                 padding='same')

    """
    architecture defined in the paper. The upsampling rate is fixed to 8 for the purpose of this project
    """

    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        xCopy = x  # skip connection
        # residual blocks
        for i in range(len(self.resBlockList)):
            x = self.resBlockList[i](x)
        x = self.conv2(x)
        x = self.batchNorm1(x)
        x = x + xCopy
        # upsampling by 4
        x = self.conv3(x)
        x = tf.nn.depth_to_space(x, block_size=4)
        x = self.prelu2(x)
        # upsampling by 2
        x = self.conv4(x)
        x = tf.nn.depth_to_space(x, block_size=2)
        x = self.prelu3(x)
        # output
        x = self.out(x)
        return x

"""
Definition of the Residual block defined in the paper 
"""


class ResidualBlock(layers.Layer):
    def __init__(self, layerNum=2, filterNum=64, kernelSize=(3, 3),
                 strideSize=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.layerNum = layerNum
        self.filterNum = filterNum
        self.convList = []
        self.batchNormList = []
        self.pReluList = []
        for i in range(layerNum):
            self.convList.append(layers.Conv2D(filters=filterNum, kernel_size=kernelSize,
                                               strides=strideSize, padding='same'))
            self.pReluList.append(layers.ReLU())
            self.batchNormList.append(layers.BatchNormalization())

    def call(self, X):
        # assert tf.shape(X)[-1] == self.filterNum, 'input channel is wrong'
        XCopy = X
        for i in range(self.layerNum):
            X = self.convList[i](X)
            X = self.batchNormList[i](X)
            if i != self.layerNum - 1:
                X = self.pReluList[i](X)
                # skip connection
        return X + XCopy