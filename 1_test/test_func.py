# coding: utf-8
import numpy as np
import SimpleITK as sitk
import math
from glob import glob
import matplotlib.pyplot as plt


def array_normalied(array):  # normalize
    img_temp = 255. / (array.max() - array.min())
    array = (array - array.min()) * img_temp
    array = np.uint8(array)
    return array


def show_image(array, color=u'gray'):  # show same images
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


def out_coord_map(old_coord, file_path, extendbox):  # voxel to word coord
    itkimage = sitk.ReadImage(file_path)
    old_spacing = np.array(list(reversed(itkimage.GetSpacing())))
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    extendbox = extendbox[:, 0]
    max_score = old_coord[:, 0].max()
    new_coord = np.copy(old_coord)

    for i in range(old_coord.shape[0]):
        # new_coord[i, 0] = old_coord[i, 0]  # score
        # new_coord[i, 0] = abs(old_coord[i, 0]) / abs(max_score)
        new_coord[i, 0] = math.e**old_coord[i, 0]/math.e**max_score
        new_coord[i, 1] = origin[2] + extendbox[2] + old_coord[i, 3]
        new_coord[i, 2] = origin[1] + extendbox[1] + old_coord[i, 2]
        new_coord[i, 3] = origin[0] + extendbox[0] + old_coord[i, 1]
        new_coord[i, 4] = old_coord[i, 4]/old_spacing[1]
    return new_coord


class GetPBB(object):  # get nodule coord
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        output_raw = np.copy(output)

        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        # x, y, z, r
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        # score
        ouput_temp = np.copy(output)
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa], ouput_temp, output_raw
            # return ouput_raw
        else:
            return output
            # conf_th = [-1, 0, 1]
            # nms_th = [0.3, 0.5, 0.7]
            # detect_th = [0.2, 0.3]
            # conf_th=0, nms_th=0.1, detect_th=0.1
            # output = output[output[:, 0] >= self.conf_th]  # mask (72, 72, 3, 5)
            # bboxes = nms(output, self.nms_th)


def nms(output, nms_th):  # get nodule and coord of center
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
        # [0, 0, 0]
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score > bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p, [bestscore]], 0))
            else:
                fp.append(np.concatenate([p, [bestscore]], 0))
        if flag == 0:
            fp.append(np.concatenate([p, [bestscore]], 0))
    for i, l in enumerate(lbb):
        if l_flag[i] == 0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5], l))
            if len(score) != 0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore < detect_th:
                fn.append(np.concatenate([l, [bestscore]], 0))

    return tp, fp, fn, len(lbb)


def topkpbb(pbb, lbb, nms_th, detect_th, topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp) + len(fp) < topk:
        conf_th = conf_th - 0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th < -3:
            break
    tp = np.array(tp).reshape([len(tp), 6])
    fp = np.array(fp).reshape([len(fp), 6])
    fn = np.array(fn).reshape([len(fn), 5])
    allp = np.concatenate([tp, fp], 0)
    sorting = np.argsort(allp[:, 0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk, len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
    #     print(fp_in_topk)
    fn_i = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i) > 0:
        fn = np.concatenate([fn, tp[fn_i, :5]])
    else:
        fn = fn
    if len(tp_in_topk) > 0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk) > 0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp, fn


def mask_path(path, filename="mask.npy", file_suffix=".npy"):  # get path
    all_file_list = glob(path + '/*' + file_suffix)
    mask_list = []
    for i in range(all_file_list.__len__()):
        if filename in all_file_list[i]:
            mask_list.append(all_file_list[i])
    return mask_list


def load_path(path, file_suffix=".mhd"):  # load file
    file_list = glob(path + "*" + file_suffix)
    if not file_list:
        file_list = glob(path + "*/*" + file_suffix)
        if not file_list:
            file_list = glob(path + "*/*/*" + file_suffix)
    return file_list
