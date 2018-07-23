import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob


def array_normalied(array):  # nomalize to 0-255
    img_temp = 255. / (array.max() - array.min())
    array = (array - array.min()) * img_temp
    array = np.uint8(array)
    return array


def show_image(array, color=u'gray'):  # show several images Simultaneously
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


output_list = glob(r'D:\ali_challenge2\1_test\1_output_test/*_clean.npy')
start = 0
for i in range(10):
    img = np.load(output_list[start + i])
    show_image(img[0, 20:36, :, :])