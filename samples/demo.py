import os
import sys
import random
import math
import numpy as np
import cv2
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import config
sys.path.append(os.path.join(ROOT_DIR, "samples/balloon/"))  # To find local version
import balloon


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0050.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def divide_method2(img, w, h, m, n):  # 分割成m行n列
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)
    plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]  #
    return divide_image


def display_blocks(divide_image):

    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(divide_image[i, j, :])
            plt.axis('off')
            plt.title('block:' + str(i * n + j + 1))


def image_concat(divide_image):
    m, n, grid_h, grid_w = [divide_image.shape[0], divide_image.shape[1],  # 每行，每列的图像块数
                            divide_image.shape[2], divide_image.shape[3]]  # 每个图像块的尺寸

    restore_image = np.zeros([m * grid_h, n * grid_w, 3], np.uint8)

    restore_image[0:grid_h, 0:]
    for i in range(m):
        for j in range(n):
            restore_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = divide_image[i, j, :]
    return restore_image

def SinglePredict(img):
    imge = input('输入原图位置:')
    # 读取原始图片路径
    img = cv2.imread(imge)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[0], img.shape[1]

    # 原始图像分块
    m = int(input('输入分块行数：'))
    n = int(input('输入分块列数：'))

    divide_image2 = divide_method2(img, w, h, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    display_blocks(divide_image2)


    plt.show()
    a = int(input('需要检测的行:'))
    b = int(input('需要检测的列:'))
    class_names = ['0', 'dong']

#    image = skimage.io.imread(divide)

    # Run detection
    results = model.detect([divide_image2[a - 1, b - 1]], verbose=1)

    # Visualize results

    r = results[0]

    # if r['class_ids'].all():
    #     mask_bool=r['class_ids']
    #     mask=mask_bool.astype(np.int)
    #     print(mask.shape)
    #     for i in range(mask.shape[2]):
    #         maski=mask[:,:,i]
    visualize.display_instances(divide_image2[a - 1, b - 1], r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

def BatchPredict(DirPath):
    fileList = os.listdir(DirPath)

    for img in fileList:
        name = DirPath + '\\' + img

        SinglePredict(name)


class InferenceConfig(balloon.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

if __name__ == "__main__":
    BatchPredict('img')
