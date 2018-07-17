import numpy as np
import matplotlib.pyplot as plt
import math
from glob import glob


def array_normalied(array):
    img_temp = 255. / (array.max() - array.min())
    array = (array - array.min()) * img_temp
    array = np.uint8(array)
    return array


def show_image(array, color=u'gray'):
    len = min(array.shape)
    axis = (array.shape).index(len)
    xx = math.ceil(np.sqrt(len))
    plt.figure()
    if axis == 0:
        for i in range(len):
            plt.subplot(xx, xx, i+1).imshow(array_normalied(array[i, :, :]), color)
    elif axis == 1:
        for i in range(len):
            plt.subplot(xx, xx, i+1).imshow(array_normalied(array[:, i, :]), color)
    elif axis == 2:
        for i in range(len):
            plt.subplot(xx, xx, i+1).imshow(array_normalied(array[:, :, i]), color)
    else:
        print("axis=0/1/2")


output_list = glob(r'D:\ali_challenge2\2_train\2_data_test/*.npy')
start = 0
for i in range(10):
    img = np.load(output_list[start + i])
    # show_image(img[30:46, :, :, 0, 0])
    # img[img < 0] = -8
    show_image(img[15:31, :, :])