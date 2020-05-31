import json, datetime, tqdm
from Utilities.lossMetric import *
"""
The training script
"""
class MinMaxGame:
    def __init__(self, generator, discriminator, extractor=None):
        self.generator = generator
        self.discriminator = discriminator
        self.extractor = extractor
        # loss logs
        self.log = dict(genMSE=[], genEntropy=[], disLoss=[], snRatio=[], genLoss=[], genContent=[], acc=[])
        # a detailed decomposition of GAN loss
        self.genLossLog = tfk.metrics.Mean(name='gen loss')
        self.mseLog = tfk.metrics.Mean(name='gen MSE')
        self.entropyLog = tfk.metrics.Mean(name='gen entropy')
        self.contentLog = tfk.metrics.Mean(name='gen content')
        self.disLossLog = tfk.metrics.Mean(name='dis loss')
        self.snRatioLog = tfk.metrics.Mean(name='Peak SN ratio')
        self.accLog = tfk.metrics.Mean(name='discriminator acc')

        # val log
        self.valLog = dict(genMSE=[], genEntropy=[], disLoss=[], snRatio=[], genLoss=[], genContent=[], acc=[])
        self.valGenLossLog = tfk.metrics.Mean(name='gen loss')
        self.valMseLog = tfk.metrics.Mean(name='gen MSE')
        self.valEntropyLog = tfk.metrics.Mean(name='gen entropy')
        self.valContentLog = tfk.metrics.Mean(name='gen content')
        self.valDisLossLog = tfk.metrics.Mean(name='dis loss')
        self.valSnRatioLog = tfk.metrics.Mean(name='Peak SN ratio')
        self.valAccLog = tfk.metrics.Mean(name='discriminator acc')

        self.now = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
    """
    The loss function has three components: 
        MSE loss, VGG loss, and the GAN loss. 
    """
    def _trainStep(self, step, lrImgs, hrImgs):
        with tf.GradientTape(persistent=True) as tape:
            fakeImgs = self.generator(lrImgs)
            fakeLogit = self.discriminator(fakeImgs)
            realLogit = self.discriminator(hrImgs)
            # losses
            disLoss = discriminatorLoss(fakeLogit, realLogit)
            disAcc = discriminatorAcc(fakeLogit, realLogit)
            mse = generatorMSE(fakeImgs, hrImgs)
            entropy = generatorGANLoss(fakeLogit)
            # content = 0 # need to be changed to VGG loss
            content = generatorPrecepLoss(self.extractor, fakeImgs, hrImgs)
            genLoss = mse + 0.2 * entropy + 0.1 * content
        snRatio = psnr(hrImgs, fakeImgs)

        if step % 1 == 0:
            genGrad = tape.gradient(genLoss, self.generator.trainable_weights)
            self.genOptimizer.apply_gradients(zip(genGrad, self.generator.trainable_weights))
        if step % 3 == 0:
            disGrad = tape.gradient(disLoss, self.discriminator.trainable_weights)
            self.disOptimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_weights))

        # record loss values
        self.mseLog(mse)
        self.entropyLog(entropy)
        self.contentLog(content)
        self.genLossLog(genLoss)
        self.disLossLog(disLoss)
        self.snRatioLog(snRatio)
        self.accLog(disAcc)

    def _valStep(self, lrImgs, hrImgs):
        fakeImgs = self.generator(lrImgs)
        fakeLogit = self.discriminator(fakeImgs)
        realLogit = self.discriminator(hrImgs)
        # losses
        disLoss = discriminatorLoss(fakeLogit, realLogit)
        disAcc = discriminatorAcc(fakeLogit, realLogit)
        mse = generatorMSE(fakeImgs, hrImgs)
        entropy = generatorGANLoss(fakeLogit)
        # content = 0 # need to be changed to VGG loss
        content = generatorPrecepLoss(self.extractor, fakeImgs, hrImgs)
        genLoss = mse + 0.05 * entropy + content
        snRatio = psnr(hrImgs, fakeImgs)
        # record loss values
        self.valMseLog(mse)
        self.valEntropyLog(entropy)
        self.valContentLog(content)
        self.valGenLossLog(genLoss)
        self.valDisLossLog(disLoss)
        self.valSnRatioLog(snRatio)
        self.valAccLog(disAcc)

    def _updateRecord(self):
        # record stats
        self.log['genMSE'].append(float(self.mseLog.result()))
        self.log['genEntropy'].append(float(self.entropyLog.result()))
        self.log['genContent'].append(float(self.contentLog.result()))
        self.log['genLoss'].append(float(self.genLossLog.result()))
        self.log['disLoss'].append(float(self.disLossLog.result()))
        self.log['snRatio'].append(float(self.snRatioLog.result()))
        self.log['acc'].append(float(self.accLog.result()))
        # reset stats
        self.genLossLog.reset_states()
        self.mseLog.reset_states()
        self.entropyLog.reset_states()
        self.contentLog.reset_states()
        self.disLossLog.reset_states()
        self.accLog.reset_states()
        self.snRatioLog.reset_states()

        # record stats
        self.valLog['genMSE'].append(float(self.valMseLog.result()))
        self.valLog['genEntropy'].append(float(self.valEntropyLog.result()))
        self.valLog['genContent'].append(float(self.valContentLog.result()))
        self.valLog['genLoss'].append(float(self.valGenLossLog.result()))
        self.valLog['disLoss'].append(float(self.valDisLossLog.result()))
        self.valLog['snRatio'].append(float(self.valSnRatioLog.result()))
        self.valLog['acc'].append(float(self.valAccLog.result()))
        # reset stats
        self.valGenLossLog.reset_states()
        self.valMseLog.reset_states()
        self.valEntropyLog.reset_states()
        self.valContentLog.reset_states()
        self.valDisLossLog.reset_states()
        self.valAccLog.reset_states()
        self.valSnRatioLog.reset_states()

        handle = open('I:/Data/log_' + self.now + '.json', 'w')
        json.dump(self.log, handle)
        handle = open('I:/Data/val_' + self.now + '.json', 'w')
        json.dump(self.valLog, handle)

    def train(self, trainData, valData, parameters):
        lrGen, lrDis = parameters['lrGenerator'], parameters['lrDiscriminator']
        epochs, stepsPerEpoch = parameters['epochs'], parameters['stepsPerEpoch']
        valSteps = parameters['valSteps']

        lrGen = tfk.optimizers.schedules.ExponentialDecay(lrGen, 8 * stepsPerEpoch, 0.2, staircase=True)
        lrDis = tfk.optimizers.schedules.ExponentialDecay(lrDis, 8 * stepsPerEpoch, 0.2, staircase=True)
        self.genOptimizer = tfk.optimizers.Adam(learning_rate=lrGen)
        self.disOptimizer = tfk.optimizers.Adam(learning_rate=lrDis)
        # training
        trainItr = iter(trainData)
        valItr = iter(valData)
        for epoch in range(epochs):
            with tqdm.tqdm_notebook(range(stepsPerEpoch)) as t:
                for i in t:
                    (lrImgs, hrImgs) = next(trainItr)
                    self._trainStep(i, lrImgs, hrImgs)
                    t.set_description('epoch {}'.format(epoch))
                    t.set_postfix(genLoss=self.genLossLog.result().numpy(),
                                  genMSE=self.mseLog.result().numpy(),
                                  genEntropy=self.entropyLog.result().numpy(),
                                  genContent=self.contentLog.result().numpy(),
                                  disLoss=self.disLossLog.result().numpy(),
                                  disAcc=self.accLog.result().numpy(),
                                  snRatio=self.snRatioLog.result().numpy())

            with tqdm.tqdm_notebook(range(valSteps)) as t:
                for i in t:
                    (lrImgs, hrImgs) = next(valItr)
                    self._valStep(lrImgs, hrImgs)
                    t.set_description('val {}'.format(epoch))
                    t.set_postfix(genLoss=self.valGenLossLog.result().numpy(),
                                  genMSE=self.valMseLog.result().numpy(),
                                  genEntropy=self.valEntropyLog.result().numpy(),
                                  genContent=self.valContentLog.result().numpy(),
                                  disLoss=self.valDisLossLog.result().numpy(),
                                  disAcc=self.valAccLog.result().numpy(),
                                  snRatio=self.valSnRatioLog.result().numpy())

            self._updateRecord()
        return self.log, self.valLog