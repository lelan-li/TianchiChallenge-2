# coding:utf-8
import numpy as np
import pandas as pd
import os
import scipy.ndimage
import warnings

from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from skimage import measure
from random import choice

from prepare_main import OneImage


def binarize_per_slice(image, spacing, intensity_th=-200, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)  # mask center is 1, other is 0
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                               truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                               truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:  # use eccentricity to remove
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw  # lunge and in center is


# def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=1e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    # check for each slice
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = {label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1],
                label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                label[-1 - cut_num, -1, -1], label[0, 0, mid], label[0, -1, mid],
                label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]}
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:  # delete small
        # if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
        if prop.area * spacing.prod() < vol_limit[0] * 1e4 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[
        1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[
        2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)  # lunge is true

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw,
        #  part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)  # lunge is 1, other is 0


def fill_hole(bw):  # fill 3d holes
    label = measure.label(~bw)
    bg_label = {label[0, 0, 0], label[0, 0, -1],
                label[0, -1, 0], label[0, -1, -1],
                label[-1, 0, 0], label[-1, 0, -1],
                label[-1, -1, 0], label[-1, -1, -1]}
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw  # lunge is true


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:  # get two lunge by area
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = (iter_count + 1)

    if found_flag:  # remove big hole
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1==False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2==False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)  # remove small hole
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def step1_python(case_path):

    itkimage = sitk.ReadImage(case_path)
    image = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    bw = binarize_per_slice(image, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return image, bw1, bw2, spacing


# -------------------------

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 2 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


id = 1


def lumTrans(img):
    # lungwin = np.array([-1200., 600.])
    lungwin = np.array([-1200., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def transform_is_right(mhd_path):
    with open(mhd_path) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

        return isflip


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def savenpy_real(id, filelist, prep_folder, data_path, label_csv_path, use_existing=True):
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder, name + '_label.npy')) and \
                os.path.exists(os.path.join(prep_folder, name + '_clean.npy')):
            print(name + ' had been done')
            return
    try:
        im, m1, m2, spacing = step1_python(data_path[id])
        oneImage = OneImage(data_path[id])
        Mask = m1 + m2
        newshape = np.round(np.array(Mask.shape) * spacing / resolution)
        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                               np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
        extendbox = extendbox.astype('int')

        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilated_mask = dm1 + dm2
        pad_value = 190

        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilated_mask + pad_value * (1 - dilated_mask).astype('uint8')  # get area of lunge
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],  # get bounding box
                            extendbox[1, 0]:extendbox[1, 1],
                            extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name + '_extendbox'), extendbox)
        np.save(os.path.join(prep_folder, name + '_clean'), sliceim)
        if label_csv_path:
            label_csv = np.array(pd.read_csv(label_csv_path))
            this_annos = np.copy(label_csv[label_csv[:, 0] == str(name)])
            # print(str(name), 'this_annoce: ', this_annos)
            label = []
            if len(this_annos) > 0:

                for c in this_annos:
                    pos = worldToVoxelCoord(c[1:4][::-1], origin=oneImage.origin(), spacing=oneImage.spacing())
                    if oneImage.isflip():
                        pos[1:] = Mask.shape[1:3] - pos[1:]
                    label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

            label = np.array(label)
            if len(label) == 0:
                label2 = np.array([[0, 0, 0, 0]])
            else:
                label2 = np.copy(label).T
                label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
                label2[3] = label2[3] * spacing[1] / resolution[1]
                label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)

                label2 = label2[:4].T
            np.save(os.path.join(prep_folder, name + '_label.npy'), label2)
        print(name + ' done')
    except:
        print('bug in ' + name)
    #     raise
    # print(name + ' done')


def savenpy(id, filelist, prep_folder, data_path, label_csv_path, use_existing=True):
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    # print(len(filelist) - id)
    if use_existing:
        if os.path.exists(os.path.join(prep_folder, name + '_label.npy')) and \
                os.path.exists(os.path.join(prep_folder, name + '_clean.npy')):
            print(name + ' had been done')
            return
    # begin
    im, m1, m2, spacing = step1_python(data_path[id])
    oneImage = OneImage(data_path[id])
    Mask = m1 + m2
    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                           np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
    extendbox = extendbox.astype('int')

    dm1 = process_mask(m1)  # into a convex hull
    dm2 = process_mask(m2)
    dilated_mask = dm1 + dm2

    pad_value = 190

    im[np.isnan(im)] = -2000
    sliceim = lumTrans(im)
    sliceim = sliceim * dilated_mask + pad_value * (1 - dilated_mask).astype('uint8')  # get the region of lung
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                        extendbox[1, 0]:extendbox[1, 1],
                        extendbox[2, 0]:extendbox[2, 1]]
    sliceim = sliceim2[np.newaxis, ...]
    np.save(os.path.join(prep_folder, name + '_extendbox'), extendbox)
    np.save(os.path.join(prep_folder, name + '_clean'), sliceim)
    if label_csv_path:
        label_csv = np.array(pd.read_csv(label_csv_path))
        this_annos = np.copy(label_csv[label_csv[:, 0] == str(name)])  # ['file_name', x,y,z]
        label = []
        if len(this_annos) > 0:

            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=oneImage.origin(), spacing=oneImage.spacing())
                if oneImage.isflip():  # mask
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)  # [x,y,z,r]
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)

            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + '_label.npy'), label2)
    print(name + ' done')


def full_prep_ali(mhd_path_list, filename_list, prep_folder, label_csv_path, n_worker=None, use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    print('starting preprocessing')

    len_temp = len(filename_list)
    file_len = list(range(len_temp))
    for i in range(len_temp):
        id = choice(file_len)
        # id = i
        file_len.remove(id)
        print(filename_list[id])
        savenpy(id=id,
                filelist=filename_list,
                prep_folder=prep_folder,
                label_csv_path=label_csv_path,
                data_path=mhd_path_list,
                use_existing=use_existing)
        print(len(filename_list) - i)
    # pool = Pool(n_worker)
    # partial_savenpy = partial(savenpy,
    #                           filelist=filename_list,
    #                           prep_folder=prep_folder,
    #                           label_csv_path=label_csv_path,
    #                           data_path=mhd_path_list,
    #                           use_existing=use_existing)
    #
    # N = len(filename_list)
    # _ = pool.map(partial_savenpy, range(N))
    # pool.close()
    # pool.join()

    print('end preprocessing')
    # return filelist



