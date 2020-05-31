import matplotlib.pyplot as plt
import numpy as np
"""
a simple script to visualize and compare the ground truth, downsampled input image, and model prediction
"""


class Visualizer:

    def plot(self, x, y, yPred=None):
        x = np.squeeze(np.clip(x, a_min=0, a_max=1))
        y = np.squeeze(np.clip(y, a_min=0, a_max=1))
        if yPred is None:
            picNum = 2
        else:
            picNum = 3
            yPred = np.squeeze(np.clip(yPred, a_min=0, a_max=1))
        fig, ax = plt.subplots(1, picNum)
        fig.set_size_inches(8, 8)
        ax[0].imshow(x)
        ax[0].set_title('input image')
        ax[1].imshow(y)
        ax[1].set_title('ground truth')
        if yPred is not None:
            ax[2].imshow(yPred)
            ax[2].set_title('recovered')