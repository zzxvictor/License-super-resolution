"""
Implementation of RRDBNet
Author: Zixuan Zhang
Date: 5/30/2020
Reference: https://arxiv.org/pdf/1809.00219.pdf
"""

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as layers

"""
implementation of the generator architecture defined in ESRGAN. 
The upsampling factor is fixed to 8 for the purpose of this project
"""


class RRDBNet(tfk.Model):
    def __init__(self, outChannel=3, blockNum=3, channel=64, increment=32, alpha=0.2):
        super(RRDBNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')
        self.lrelu = layers.LeakyReLU(alpha=alpha)
        self.rrdb = RRDB(blockNum, channel, increment, alpha)
        self.conv2 = layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=channel * 128, kernel_size=3, strides=1, padding='same')
        self.convOut = layers.Conv2D(filters=outChannel, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        x = self.conv1(x)  # high level features
        xCopy = x
        x = self.rrdb(x)
        x = self.conv2(x)
        x = x + xCopy
        #  first upsampling
        x = self.conv3(x)
        x = tf.nn.depth_to_space(x, block_size=4)  # channel*128 -> channel*16
        x = self.lrelu(x)
        #  second upsampling
        x = tf.nn.depth_to_space(x, block_size=2)  # channel*16 -> channel*4
        x = self.lrelu(x)
        # final output
        x = self.convOut(x)
        return x

"""
Residual in residual dense block definition 
"""


class RRDenseBlock(layers.Layer):
    def __init__(self, channel=64, increment=32, alpha=0.2):
        super(RRDenseBlock, self).__init__()
        self.alpha = alpha
        self.conv1 = layers.Conv2D(filters=increment, kernel_size=3, strides=1, padding='same')
        self.conv2 = layers.Conv2D(filters=increment, kernel_size=3, strides=1, padding='same')
        self.conv3 = layers.Conv2D(filters=increment, kernel_size=3, strides=1, padding='same')
        self.conv4 = layers.Conv2D(filters=increment, kernel_size=3, strides=1, padding='same')
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.convOut = layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(tf.concat([x, x1], axis=-1)))
        x3 = self.lrelu(self.conv3(tf.concat([x, x1, x2], axis=-1)))
        x4 = self.lrelu(self.conv4(tf.concat([x, x1, x2, x3], axis=-1)))
        out = self.convOut(tf.concat([x, x1, x2, x3, x4], axis=-1))
        return out * self.alpha + x


'''
a sequence of RRDenseBlock
'''


class RRDB(layers.Layer):
    def __init__(self, blockNum=3, channel=64, increment=32, alpha=0.2):
        super(RRDB, self).__init__()
        self.blockList = []
        self.alpha = alpha
        for i in range(blockNum):
            self.blockList.append(RRDenseBlock(channel, increment, alpha))

    def call(self, x):
        xCopy = x
        for block in self.blockList:
            x = block(x)
        return xCopy * self.alpha + x
