import numpy as np
import pandas as pd
import SimpleITK as sitk
import math
import scipy

from glob import glob
from matplotlib import pyplot as plt
from skimage import  morphology, segmentation


from prepare_func import *
from prepare_config import prepare_config


class OneImage:  # process at each CT data
    def __init__(self, path):
        self.path = path
        self.file_name = path[-14:-4]

    def all_massage(self):
        itk_img = sitk.ReadImage(self.path)
        image = sitk.GetArrayFromImage(itk_img)
        spacing = np.array(list(reversed(itk_img.GetSpacing())))
        origin = np.array(list(reversed(itk_img.GetOrigin())))
        return image, spacing, origin

    def image(self):  # get all pixel
        itk_img = sitk.ReadImage(self.path)
        image = sitk.GetArrayFromImage(itk_img)
        return image

    def spacing(self):  # get voxel
        itk_img = sitk.ReadImage(self.path)
        spacing = np.array(list(reversed(itk_img.GetSpacing())))
        return spacing

    def origin(self):  # get origin
        itk_img = sitk.ReadImage(self.path)
        origin = np.array(list(reversed(itk_img.GetOrigin())))
        return origin

    def label(self):
        pass

    def save_prepared_image(self):
        pass

    def isflip(self):
        with open(self.path) as f:
            contents = f.readlines()
            line = [k for k in contents if k.startswith('TransformMatrix')][0]
            transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
            transform = np.round(transform)
            if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
                isflip = True
            else:
                isflip = False

            return isflip

    def imshow(self, image, color=u'gray'):
        len = min(image.shape)
        axis = (image.shape).index(len)
        xx = math.ceil(np.sqrt(len))
        plt.figure()
        if axis == 0:
            for i in range(len):
                plt.subplot(xx, xx, i + 1).imshow(self.image_0_255(image[i, :, :]), color)
        elif axis == 1:
            for i in range(len):
                plt.subplot(xx, xx, i + 1).imshow(self.image_0_255(image[:, i, :]), color)
        elif axis == 2:
            for i in range(len):
                plt.subplot(xx, xx, i + 1).imshow(self.image_0_255(image[:, :, i]), color)
        else:
            print("axis=0/1/2")

    def image__1000_400(self, image_raw, min_bound=-1000.0, max_bound=400.0, bone=100, meat=-67):
        image_raw[image_raw > max_bound] = max_bound
        image_raw[image_raw < min_bound] = min_bound
        image_raw[image_raw > bone] = meat
        return image_raw

    def image_0_255(self, image):  # normalize
        img_temp = 255. / (image.max() - image.min())
        image = (image - image.min()) * img_temp
        image = np.uint8(image)
        return image

    def resample(self, image, old_spacing, new_spacing=[1, 1, 1]):  # resample
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor

        image_resample = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image_resample, new_spacing

    def segmentate_lung(self, image, threshold=140, square_one=5, squre_two=7, disk_two=9):
        image_temp = image.copy()
        image_temp[image_temp <= threshold] = 0
        image_temp[image_temp > threshold] = 1
        image_temp = 1 - image_temp
        if image_temp.min() < 0 or image_temp.max() > 1:

        for i in range(image_temp.shape[0]):
            image_temp[i] = morphology.binary_dilation(image_temp[i], morphology.square(square_one))  # ����
            image_temp[i] = morphology.binary_erosion(image_temp[i], morphology.square(squre_two))
            image_temp[i] = segmentation.clear_border(image_temp[i])
            # image_temp[i] = morphology.binary_dilation(image_temp[i], morphology.disk(disk_two))
        return image_temp

    def main(self, low_slice=100, high_slice=116):
        image = self.image_0_255(self.image__1000_400(self.image()))
        image_temp = self.segmentate_lung(image)
        self.imshow(image_temp[low_slice:high_slice, :, :])


# preprocess at all the CT data
class AllImage:
    def __init__(self, prepare_config, process='test2'):
        self.prepare_config = prepare_config
        self.process = process

        self.data_folder = self.prepare_config['data_folder' + '_' + process]
        self.code_folder = self.prepare_config['code_folder']

        self.all_mhd_path_list = self.all_mhd_path_list()
        self.all_filename_list = self.all_filename_list()
        self.done_mhd_path_list = self.done_mhd_path_list()
        self.done_filename_list = self.done_filename_list()
        self.remain_mhd_path_list = self.remain_mhd_path_list()
        self.remain_filename_list = self.remain_filename_list()

        if process == 'train' or process == 'val':
            self.label_csv_path = prepare_config['annotations_path_' + process]
        else:
            self.label_csv_path = []

    def __len__(self):
        return len(self.all_mhd_path_list)

    def __getitem__(self, item):
        return OneImage(self.all_mhd_path_list[item])

    def csv(self):
        csv_temp = pd.read_csv(self.prepare_config['annotations_path'] + self.process)
        return csv_temp

    def all_mhd_path_list(self, file_suffix=".mhd"):
        file_list = glob(self.data_folder + "/*" + file_suffix)
        if not file_list:
            file_list = glob(self.data_folder + "/*/*" + file_suffix)
            if not file_list:
                file_list = glob(self.data_folder + "/*/*/*" + file_suffix)
        return file_list

    def all_filename_list(self):
        filename_list_temp = []
        for i in self.all_mhd_path_list:
            filename_list_temp.append(i[-14:-4])
        return filename_list_temp

    def done_mhd_path_list(self):
        return glob(self.code_folder + '/prepare' +
                    self.prepare_config['prepare_result_' + self.process] + '/*_clean.npy')

    def done_filename_list(self):
        filename_list_temp = []
        for i in self.done_mhd_path_list:
            filename_list_temp.append(i[-20:-10])
        return filename_list_temp

    def remain_mhd_path_list(self):
        remain_mhd_path_list = []
        for i in self.all_mhd_path_list:
            i_name = i[-14:-4]
            exist = False
            for j in self.done_filename_list:
                if i_name in j:
                    exist = True
            if exist is False:
                remain_mhd_path_list.append(i)
        return remain_mhd_path_list

    def remain_filename_list(self):
        filename_list_temp = []
        for i in self.remain_mhd_path_list:
            filename_list_temp.append(i[-14:-4])
        return filename_list_temp

    def main(self):
        full_prep_ali(self.remain_mhd_path_list,
                      filename_list=self.remain_filename_list,
                      prep_folder=self.prepare_config['prepare_result_' + self.process],
                      n_worker=self.prepare_config['n_worker_preprocessing'],
                      use_existing=self.prepare_config['use_exsiting_preprocessing'],
                      label_csv_path=self.label_csv_path)


# check the data after preprocess
class CheckImage:
    def __init__(self, prepare_config, process='test2'):
        self.process = process
        self.prepare_config = prepare_config
        self.prepare_check = self.prepare_config['prepare_check_' + self.process]
        self.prepare_data_folder = self.prepare_config['prepare_result_' + self.process]
        self.clean_path_list = self.clean_path_list()
        self.extendbox_path_list = self.extendbox_path_list()

    def clean_path_list(self):
        return glob(self.prepare_data_folder + '/*_clean.npy')

    def extendbox_path_list(self):
        return glob(self.prepare_data_folder + '/*_extendbox.npy')

    def save_shape(self):
        shape = {'seriesuid': [], 'X': [], 'Y': [], 'Z': []}
        for i in self.clean_path_list:
            shape_temp = np.load(i).shape
            shape['seriesuid'].append(i[-20:-10])
            shape['X'].append(shape_temp[1])
            shape['Y'].append(shape_temp[2])
            shape['Z'].append(shape_temp[3])

        if shape['seriesuid'].__len__() > 0:
            df = pd.DataFrame(shape)
            df.to_csv(self.prepare_check + '/' + self.process + '_shape.csv', index=False)

    def save_extendbox(self):
        extendox = {'seriesuid': [], 'LX': [], 'LY': [], 'LZ': [], 'HX': [], 'HY': [], 'HZ': []}
        for i in self.extendbox_path_list:
            extendox['seriesuid'].append(i[-24:-14])
            i_coord = np.load(i)
            extendox['LX'].append(i_coord[0, 0])
            extendox['LY'].append(i_coord[1, 0])
            extendox['LZ'].append(i_coord[2, 0])
            extendox['HX'].append(i_coord[0, 1])
            extendox['HY'].append(i_coord[1, 1])
            extendox['HZ'].append(i_coord[2, 1])

        if extendox['seriesuid'].__len__() > 0:
            df = pd.DataFrame(extendox)
            df.to_csv(self.prepare_check + '/' + self.process + '_extenbox.csv', index=False)

    def save_image(self):
        for i in self.clean_path_list:
            seriesuid = i[-20:-10]
            image = np.load(i)
            shape = image.shape
            print(seriesuid, shape)
            img0 = image[0, round(shape[1] / 3), :, :]
            img1 = image[0, round(shape[1] / 2), :, :]
            img2 = image[0, round(shape[1]*2 / 3), :, :]
            print(round(shape[1] / 3), round(shape[1] / 2), round(shape[1]*2 / 3))
            scipy.misc.imsave(self.prepare_check + "/%s_%.2f.jpg" % (seriesuid, 0.33), img0)
            scipy.misc.imsave(self.prepare_check + "/%s_%.2f.jpg" % (seriesuid, 0.50), img1)
            scipy.misc.imsave(self.prepare_check + "/%s_%.2f.jpg" % (seriesuid, 0.66), img2)

    def main(self):
        self.save_shape()
        self.save_extendbox()
        self.save_image()


if __name__ == '__main__':
    for process in ['train', 'val', 'test']:
        allImage = AllImage(prepare_config, process)
        allImage.main()
        checkeImage = CheckImage(prepare_config, process)
        checkeImage.main()