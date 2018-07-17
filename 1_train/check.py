import numpy as np
import data
from train_config import train_config as config
import random
import math
import matplotlib.pyplot as plt

run = True


# functions
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


# load data
dataset_train = data.DataBowl3Detector(config, process='all')
for i in range(20000, 20100):
    x, y, z = dataset_train[i]

# check dataset
n = 100  # 3750
for i in range(10):
    bbox = dataset_train.bboxes[n + 8*i]
    filename = dataset_train.filenames[int(bbox[0])]
    print(i, filename)
    x, y, z = dataset_train[n + i]
    coord = np.where(y[:, 0, ...] >= 0.5)
    if len(coord[0]) > 0:
        show_image(y[coord[0][0], 0, coord[1][0] - 1: coord[1][0] + 2, ...])
        show_image(x[0, 4 * coord[1][0]-2: 4 * coord[1][0]+2, ...])


n = 100  # 3750
for i in range(10):
    x, y, _ = dataset_train[n + i]
    show_image(x[0, 60:76, ...])
    show_image(y[0, 0, 15:19, ...])

# check random samples and shape
from glob import glob
import numpy as np
num_list = glob(r'D:\ali_challenge2\2_train\2_data_train/*.npy')
n = 100
for i in range(10):
    img = np.load(num_list[n + i])
    show_image(img[20:36, ...])

bbox = dataset_train.bboxes[1030]
filename = dataset_train.filenames[int(bbox[0])]
print(i, filename)
x, y, z = dataset_train[n]
coord = np.where(y[:, 0, ...] >= 0.5)
if len(coord[0]) > 0:
    show_image(y[coord[0][0], 0, coord[1][0] - 1: coord[1][0] + 2, ...])
    show_image(x[0, 4 * coord[1][0]-8: 4 * coord[1][0]+9, ...])
    print(x.shape, y.shape, z.shape)

bad = []
nodules_len = 2750
for i in range(nodules_len):
    try:
        x, y, z = dataset_train[i]
    except:
        bad.append(i)
        print('bad_image: ', i)
        x, y, z = dataset_train[random.sample(range(nodules_len), 1)[0]]
    print(i)
    if x.shape != (1, 128, 128, 128):
        bad.append(i)
        bad.append(x.shape)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print('\n')

print('bad: ', bad)