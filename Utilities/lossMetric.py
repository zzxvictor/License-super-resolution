import tensorflow as tf
import tensorflow.keras as tfk
"""
Evaluation metrics 
"""


def psnr(y_true, y_pred):
    mse = tf.reduce_mean(tfk.losses.MSE(y_true, y_pred))
    psnr = 10*tf.math.log(1.0/mse)
    log10 = tf.math.log(tf.constant(10, dtype=psnr.dtype))
    return psnr/log10


def ssim(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val = 1.0)
    return tf.reduce_mean(ssim)


'''
VGG feature extractor, used to calculate the vgg loss 
reference: https://arxiv.org/pdf/1609.04802.pdf
'''


def buildExtractor():
    vgg = tf.keras.applications.VGG19(include_top=False)
    config = vgg.get_layer('block2_conv2').get_config()
    config['activation'] = tf.keras.activations.linear
    config['name'] = 'output'
    output = tf.keras.layers.Conv2D(**config)(vgg.get_layer('block2_conv2').output)
    extractor = tf.keras.Model(inputs=vgg.input, outputs=output)
    extractor.layers[-1].set_weights(vgg.get_layer('block2_conv2').get_weights())
    return extractor

"""
loss function for the generator and the discriminator
"""

def discriminatorLoss(fakeLogit, realLogit):
    fakeLoss = tf.reduce_mean(tfk.losses.binary_crossentropy(y_pred=fakeLogit, y_true=tf.zeros_like(fakeLogit),
                                                             from_logits=True))
    realLoss = tf.reduce_mean(tfk.losses.binary_crossentropy(y_pred=realLogit, y_true=tf.ones_like(realLogit),
                                                             from_logits=True))
    return fakeLoss + realLoss


def generatorGANLoss(fakeLogit):
    return tf.reduce_mean(tfk.losses.binary_crossentropy(y_pred=fakeLogit, y_true=tf.ones_like(fakeLogit),
                                                         from_logits=True))


def generatorMSE(hrPred, hrImgs):
    return tf.reduce_mean(tfk.losses.mse(y_pred=hrPred, y_true=hrImgs))


def discriminatorAcc(fakeLogit, realLogit):
    accReal = tfk.metrics.binary_accuracy(tf.ones_like(realLogit), tf.math.sigmoid(realLogit))
    accFake = tfk.metrics.binary_accuracy(tf.zeros_like(fakeLogit), tf.math.sigmoid(fakeLogit))
    return (tf.reduce_mean(accReal) + tf.reduce_mean(accFake))/2


def generatorPrecepLoss(extractor, hrPred, hrImgs):
    featurePred = extractor(hrPred)
    feature = extractor(hrImgs)
    return tf.reduce_mean(tfk.losses.mse(featurePred, feature))

