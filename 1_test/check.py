import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
run = True


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


output_list = glob(r'D:\ali_challenge2\1_test\1_output_test/*output_recover.npy')
start = 0
for i in range(10):
    output_recover = np.load(output_list[start + i])
    [coord_z, coord_x, coord_y] = output_recover.shape[:3]
    for index_x in range(coord_x):
        x_base = output_recover[int(coord_z / 2), - 1, :, :, 0].mean()
        if output_recover[int(coord_z / 2), coord_x - index_x - 1, :, :, 0].mean() > 0.93 * x_base:
            x_wide = int((coord_x - index_x) / 2)
            break
    for index_y in range(coord_y):
        y_base = output_recover[int(coord_z / 2), :, - 1, :, 0].mean()
        if output_recover[int(coord_z / 2), :, coord_y - index_y - 1, :, 0].mean() > 0.93 * y_base:
            y_wide = int((coord_y - index_y) / 2)
            break
    print(0, [coord_z, coord_x, coord_y])
    print(1, x_wide, y_wide)
    r_wide = ((x_wide + y_wide) / 3.3) ** 2
    print(-1, x_base, y_base, r_wide)
    for x in range(output_recover.shape[1]):
        for y in range(output_recover.shape[2]):
            for z in range(output_recover.shape[0]):
                if (x - x_wide) ** 2 + (y - y_wide) ** 2 * 0.8 <= r_wide * (1 - np.fabs(z - coord_z / 2) / coord_z * 0.4):
                    output_recover[z, x, y, 0, 0] = -1
                    output_recover[z, x, y, 1, 0] = -1
                if (x - x_wide) ** 2 + (y - y_wide) ** 2 * 0.8 <= 1.1 * r_wide * (
                    1 - np.fabs(z - coord_z / 2) / coord_z * 0.4):
                    output_recover[z, x, y, 2, 0] = -1
    img = output_recover
    img[img[..., 0, 0] < img[..., 0, 0].max()*0.7] -= 8
    print(img[..., 0, 0].max()*0.8)
    show_image(img[18:54, :, :, 0, 0])

output_list = glob(r'D:\ali_challenge2\1_test\1_output_test/*output_recover.npy')
start = 0
for i in range(10):
    output_recover = np.load(output_list[start + i])
    img = output_recover
    img[img[..., 0, 0] < img[..., 0, 0].max()*0.7] -= 8
    print(img[..., 0, 0].max()*0.8)
    show_image(img[20:36, :, :, 0, 0])