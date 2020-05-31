import tensorflow.keras as tfk
import tensorflow.keras.layers as layers
"""
A simple VGG-style classification network used as discriminator
Author: Zixuan Zhang
Date: 5/30/2020
Reference: https://arxiv.org/pdf/1609.04802.pdf
"""
'''
discriminator
'''


class Discriminator(tfk.Model):
    def __init__(self, initChannel=64, layerNum=5):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same')
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.convBlocks = []
        for i in range(1, layerNum + 1):
            self.convBlocks.append(ConvBlock(filterNum=initChannel * i, kernelSize=2, strideSize=1, padding='same'))
            self.convBlocks.append(ConvBlock(filterNum=initChannel * i, kernelSize=2, strideSize=2, padding='same'))

        # self.flatten = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(1)  # output is logit

    def call(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        for convblock in self.convBlocks:
            x = convblock(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.lrelu(x)
        x = self.dense2(x)
        return x

"""
A sequence of convolution, batch norm, activation layers 
"""
class ConvBlock(layers.Layer):
    def __init__(self, filterNum=128, kernelSize=(3,3), strideSize=(1,1), alpha=0.2, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=filterNum, kernel_size=kernelSize, strides=strideSize,
                                   padding=padding)
        self.batchNorm = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(alpha=alpha)

    def call(self, X):
        X = self.conv1(X)
        X = self.batchNorm(X)
        return self.lrelu(X)