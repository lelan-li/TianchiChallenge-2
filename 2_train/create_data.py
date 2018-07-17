# encoding: utf-8
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
from train_config import train_config as config


class nodules_crop(object):
    def __init__(self, config, process='train'):
        self.process = process
        self.save_data_path = config['save_data_' + process]

        self.ls_all_patients = glob(config['data_folder_' + process] + "*.mhd")
        self.df_candidates = pd.read_csv(config['csv_' + process])
        self.df_candidates["file"] = \
            self.df_candidates["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_candidates = self.df_candidates.dropna()

    def img_normalize(self, image, img_min=-1200., img_max=400.):
        image[image < img_min] = img_min
        image[image > img_max] = img_max
        image = image*255./(400+1200)
        return image

    def resample(self, image, old_spacing, new_spacing=[1, 1, 1]):
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing

    def get_filename(self, file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def candidates_crop(self):
        n = len(self.ls_all_patients)
        print("总共:", n)
        for patient in enumerate(self.ls_all_patients):
            n = (n - 1)
            patient = patient[1]
            print(patient)
            # nodule big than 3mm or not
            if patient not in self.df_candidates.file.values:
                print('Patient ' + patient + 'Not exist!')
                continue
            patient_nodules = self.df_candidates[self.df_candidates.file == patient]
            full_image_info = sitk.ReadImage(patient)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  # get coord of the center
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]
            image, new_spacing = self.resample(full_scan, old_spacing)  # resample
            print(0, image.shape)
            print('Resample Done')

            for index, nodule in patient_nodules.iterrows():
                nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
                print(3, nodule_center)  # word space
                v_center = np.rint((nodule_center - origin) / new_spacing)  # map to voxel space
                v_center = np.array(v_center, dtype=int)
                print(1, v_center)

                window_size = 24  # window_size+window_size
                zyx_1 = v_center - window_size  # z,y,x
                zyx_2 = v_center + window_size

                try:
                    img_crop = np.asarray(image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]])
                    img_crop = self.img_normalize(img_crop)  # set windows width
                    print("img_crop:", img_crop[1, 1, 20:25], img_crop.shape)
                    if self.process != 'test':
                        np.save(config['save_data_' + self.process] + str(nodule['seriesuid']) + "_X" + str(round(nodule['coordX'], 3))
                                + '_Y' + str(round(nodule['coordY'], 3)) + '_Z' + str(round(nodule['coordZ'], 3)) +
                                '_R' + str(round(nodule['diameter_mm'], 3)) + '.npy', img_crop)
                    else:
                        np.save(config['save_data_' + self.process] + str(nodule['seriesuid']) + "_X" + str(round(nodule['coordX'], 3))
                                + '_Y' + str(round(nodule['coordY'], 3)) + '_Z' + str(round(nodule['coordZ'], 3)) +
                                '_P' + str(round(nodule['probability'], 3)) + '.npy', img_crop)
                except Exception as e:
                    print(Exception, ': ', e)
                    print(index, "error!")
                    continue
                print('Done for this patient!\n')
            print('Done for all!')
            print("Remain:", n)


if __name__ == '__main__':
    for process in ['train', 'val', 'test']:
        nc = nodules_crop(config, process)
        nc.candidates_crop()
