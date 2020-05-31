import tensorflow as tf
import tensorflow.keras as tfk

"""
Implementation of Efficient Subpixel Convolution Network (ESPCN)
Author: Zixuan Zhang 
Date: 5/30/2020
Reference: https://arxiv.org/abs/1609.05158
"""


class ESPCN(tfk.Model):
    def __init__(self, inputShape, upScale):
        super(ESPCN, self).__init__()
        self.inputShape = inputShape
        self.upScale = upScale
        self._buildNetwork()

    """
    definition of network, the exact number of conv layers may differ from the original definition in the paper 
    """
    def _buildNetwork(self):
        self.conv1 = tfk.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')
        self.conv1 = tfk.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')
        self.conv2 = tfk.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv2 = tfk.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.conv3 = tfk.layers.Conv2D(filters=self.inputShape[-1]*self.upScale**2, kernel_size=(3, 3),
                                       padding='same', activation='linear')

    def call(self, image):
        y = self.conv1(image)
        y = self.conv2(y)
        y = self.conv3(y)
        # subpixel convolution
        y = tf.nn.depth_to_space(y, block_size=self.upScale, )
        return y

