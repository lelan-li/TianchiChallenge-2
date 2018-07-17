import numpy as np
import os
from glob import glob


class DataBowl3Detector:  # generate data:[3,5,208,208,208],coord:[3,52,52,52]
    def __init__(self, config, process='test', split_comber=None):
        self.data_dir = config['preprocess_result_' + process]
        self.process = process  # 'test'
        self.max_stride = config['max_stride']  # 16
        self.stride = config['stride']  # 4

        self.blacklist = config['blacklist']

        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']  # 170

        self.split_comber = split_comber
        self.all_name_path = self.all_name_path()
        self.all_image_name = self.all_filename_list()

        self.channel = config['chanel']
        self.filenames = [os.path.join(self.data_dir, '%s_clean.npy' % idx) for idx in self.all_image_name]
        self.has_done = glob(config['1_test_csv_' + process] + '*pbb.npy')[-18:-9]

    def __getitem__(self, idx, split=None):
        imgs = np.load(self.filenames[idx])
        nz, nh, nw = imgs.shape[1:]
        pz = int(np.ceil(float(nz) / self.stride)) * self.stride
        ph = int(np.ceil(float(nh) / self.stride)) * self.stride
        pw = int(np.ceil(float(nw) / self.stride)) * self.stride
        imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                      constant_values=self.pad_value)
        xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                 np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                 np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
        imgs, nzhw = self.split_comber.split(imgs)
        coord2, nzhw2 = self.split_comber.split(coord,
                                                side_len=self.split_comber.side_len / self.stride,
                                                max_stride=self.split_comber.max_stride / self.stride,
                                                margin=self.split_comber.margin / self.stride)
        assert np.all(nzhw == nzhw2)
        imgs = (imgs.astype(np.float32) - 128) / 128
        return np.asarray(imgs, dtype=np.float32), np.asarray(coord2, dtype=np.float32), np.array(nzhw)

    def __len__(self):
        return len(self.all_filename_list())

    def all_name_path(self, file_suffix="clean.npy"):  # get all path
        file_list = glob(self.data_dir + "/*" + file_suffix)
        if not file_list:
            file_list = glob(self.data_dir + "/*/*" + file_suffix)
            if not file_list:
                file_list = glob(self.data_dir + "/*/*/*" + file_suffix)
        file_list = [x[:-10] for x in file_list]
        return file_list

    def all_filename_list(self):  # get all name
        filename_list_temp = [x[-10:] for x in self.all_name_path]
        return filename_list_temp