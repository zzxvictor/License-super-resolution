import tensorflow as tf
"""
Scripts that load images from disk
"""


class DataLoader:
    def __init__(self, scale=8):
        self.scale = scale
    """
    Load an image from disk and return it as a tensor
    """
    def decodeImg(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        #image = tf.image.rgb_to_grayscale(image)
        #image = tf.image.adjust_contrast(image, 0.7)
        return image, tf.shape(image)
    """
    preprocess the image to create labels 
    """
    def processXY(self, path):
        y, size = self.decodeImg(path)
        x = tf.image.resize(y, [size[0]/self.scale, size[1]/self.scale])
        x = tf.image.random_brightness(x, 0.3)  # random brightness
        x = tf.image.random_contrast(x,0.5, 2)  # random contrast
        return x, y

    def load(self, fileList, batchSize=24):
        fileList = tf.data.Dataset.from_tensor_slices(fileList)
        data = fileList.map(self.processXY, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # repeat forever, requires explicit epoch control in the training script
        data = data.batch(batchSize).repeat()
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data
