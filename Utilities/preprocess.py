import numpy as np
import os
import matplotlib.image as mpimage
import argparse
from dask.distributed import LocalCluster
from dask import bag as dbag
from dask.diagnostics import ProgressBar
from typing import Tuple
from PIL import Image


'''
    crop license plates from stree views and resize all plates to the same shape 
    
'''
# Dataset statistics that I gathered in development
# This can be used to partially filter out bad images with low perceptual quality
IMAGE_MEAN = 0.5
IMAGE_MEAN_STD = 0.028

IMG_STD = 0.28
IMG_STD_STD = 0.01


def readImage(fileName: str) -> np.ndarray:
    image = mpimage.imread(fileName)
    return image


'''
    extract license plate coordinates from file name
'''


def parseLabel(label: str) -> Tuple[np.ndarray, np.ndarray]:
    annotation = label.split('-')[3].split('_')
    coor1 = [int(i) for i in annotation[0].split('&')]
    coor2 = [int(i) for i in annotation[1].split('&')]
    coor3 = [int(i) for i in annotation[2].split('&')]
    coor4 = [int(i) for i in annotation[3].split('&')]
    coor = np.array([coor1, coor2, coor3, coor4])
    center = np.mean(coor, axis=0)
    return coor, center.astype(int)


'''
    crop the images to extract license plate 
'''


def cropImage(image: np.ndarray, coor: np.ndarray, center: np.ndarray) -> np.ndarray:
    maxW = np.max(coor[:, 0] - center[0])  # max plate width
    maxH = np.max(coor[:, 1] - center[1])  # max plate height

    xWanted = [64, 128, 192, 256]
    yWanted = [32, 64, 96, 128]

    found = False
    for w, h in zip(xWanted, yWanted):
        if maxW < w//2 and maxH < h//2:
            maxH = h//2
            maxW = w//2
            found = True
            break
    if not found:  # plate too large, discard
        return np.array([])
    elif center[1]-maxH < 0 or center[1]+maxH >= image.shape[1] or \
            center[0]-maxW < 0 or center[0] + maxW >= image.shape[0]:
        return np.array([])
    else:
        return image[center[1]-maxH:center[1]+maxH, center[0]-maxW:center[0]+maxW]


'''
    save license plate 
'''


def saveImage(image: np.ndarray, fileName: str, outDir: str) -> int:
    if image.shape[0] == 0:
        return 0
    else:
        imgShape = image.shape
        if imgShape[1] == 64:
            mpimage.imsave(os.path.join(outDir, '64_32', fileName), image)
        elif imgShape[1] == 128:
            mpimage.imsave(os.path.join(outDir, '128_64', fileName), image)
        elif imgShape[1] == 208:
            mpimage.imsave(os.path.join(outDir, '192_96', fileName), image)
        else: #resize large images
            image = Image.fromarray(image).resize((192, 96))
            image = np.asarray(image) # back to numpy array
            mpimage.imsave(os.path.join(outDir, '192_96', fileName), image)
        return 1

'''
    wrap the pipeline into one function so that the processing job can be mapped to each partitions
'''


def processImage(file: str, inputDir: str, outputDir: str, subFolder: str) -> int:
    result = parseLabel(file)
    filePath = os.path.join(inputDir,subFolder, file)
    image = readImage(filePath)
    plate = cropImage(image, result[0], result[1])
    if plate.shape[0] == 0:
        return 0
    mean = np.mean(plate/255.0)
    std = np.std(plate/255.0)
    # bad brightness
    if mean <= IMAGE_MEAN - 10*IMAGE_MEAN_STD or mean >= IMAGE_MEAN + 10*IMAGE_MEAN_STD:
        return 0
    # low contrast
    if std <= IMG_STD - 10*IMG_STD_STD:
        return 0
    status = saveImage(plate, file, outputDir)
    return status


def main(argv):
    jobNum = int(argv.jobNum)
    outputDir = argv.outputDir
    inputDir = argv.inputDir
    try:
        os.mkdir(outputDir)
        for shape in ['64_32', '128_64', '192_96']:
            os.mkdir(os.path.join(outputDir, shape))
    except OSError:
        pass  # path already exists
    client = LocalCluster(n_workers=jobNum, threads_per_worker=5)  # IO intensive, more threads
    print('* number of workers:{}, \n* input dir:{}, \n* output dir:{}\n\n'.format(jobNum, inputDir, outputDir))
    #print('* Link to local cluster dashboard: ', client.dashboard_link)
    for subFolder in ['ccpd_base', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather']:
        fileList = os.listdir(os.path.join(inputDir, subFolder))
        print('* {} images found in {}. Start processing ...'.format(len(fileList), subFolder))
        toDo = dbag.from_sequence(fileList, npartitions=jobNum*30).persist()  # persist the bag in memory
        toDo = toDo.map(processImage, inputDir, outputDir, subFolder)
        pbar = ProgressBar(minimum=2.0)
        pbar.register()  # register all computations for better tracking
        result = toDo.compute()
        print('* image cropped: {}. Done ...'.format(sum(result)))
    client.close()  # shut down the cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('jobNum', help='number of worker to use', type=int)
    parser.add_argument('inputDir', help='input image directory', type=str)
    parser.add_argument('outputDir', help='output directory', type=str)
    args = parser.parse_args()
    main(args)
