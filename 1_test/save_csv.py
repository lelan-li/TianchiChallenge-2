import numpy as np
import pandas as pd
import os
import time
from glob import glob
from test_func import mask_path, nms, out_coord_map, load_path
from test_config import test_config as config


class SaveCsv:
    def __init__(self, config, process='test'):
        self.process = process
        self.true_bbox_folder = config['true_bbox_folder_' + process]
        self.extendbox_folder = config['preprocess_result_' + process]
        self.save_csv_dir = config['save_csv_dir']
        self.mhd_path_list = load_path(config['data_folder_' + process], '.mhd')

        self.true_bbox_list = glob(self.true_bbox_folder + '*pbb.npy')
        self.extendbox_list = glob(self.extendbox_folder + '*extendbox.npy')
        self.true_bbox_result = {'seriesuid': [],
                                 'coordX': [],
                                 'coordY': [],
                                 'coordZ': [],
                                 # 'diameter_mm': [],
                                 'probability': []}
        self.num_bbox = len(self.true_bbox_list)
        self.output_recover_list = glob(self.true_bbox_folder + '*_output_recover.npy')

    def __call__(self, conf_th, nms_th, detect_th):
        num_bbox = self.num_bbox
        # for i in range(1):
        for i in range(self.num_bbox):
            """
            print('start', self.true_bbox_list[i][-18:-8])
            bbox_temp = np.load(self.true_bbox_list[i])
            # bbox_temp = bbox_temp[bbox_temp[:, 0] >= conf_th  
            bbox_temp = bbox_temp[bbox_temp[:, 0] >= bbox_temp[:, 0].max()*0.7]  
            """
            print(0, self.output_recover_list[i])
            output_recover = np.load(self.output_recover_list[i])

            [coord_z, coord_x, coord_y] = output_recover.shape[:3]  # 将肺部中心误检测去掉
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
            r_wide = ((x_wide + y_wide) / 3.3) ** 2
            for x in range(output_recover.shape[1]):
                for y in range(output_recover.shape[2]):
                    for z in range(output_recover.shape[0]):
                        if (x - x_wide) ** 2 + (y - y_wide) ** 2 * 0.8 <= r_wide * (1 - np.fabs(z - coord_z / 2) / coord_z * 0.4):
                            output_recover[z, x, y, 0, 0] = -1
                            output_recover[z, x, y, 1, 0] = -1
                        if (x - x_wide) ** 2 + (y - y_wide) ** 2 * 0.8 <= 1.1*r_wide * (1 - np.fabs(z - coord_z / 2) / coord_z * 0.4):
                            output_recover[z, x, y, 2, 0] = -1

            mask = np.zeros_like(output_recover[..., 0])
            mask = np.asarray(mask, dtype=bool)
            mask0 = output_recover[..., 0, 0] > max(0, output_recover[..., 0, 0].max()*0.7)
            mask1 = output_recover[..., 1, 0] > max(0, output_recover[..., 1, 0].max()*0.8)
            mask2 = output_recover[..., 2, 0] > max(1.5, output_recover[..., 2, 0].max()*0.7)
            mask[..., 0] = mask0
            mask[..., 1] = mask1
            mask[..., 2] = mask2
            xx, yy, zz, aa = np.where(mask)
            bbox_temp = output_recover[xx, yy, zz, aa]

            print(1, bbox_temp.shape[0])
            bbox_temp = nms(bbox_temp, nms_th)
            print(2, bbox_temp.shape[0])
            bbox_temp_path = []
            skip_append = False

            extendbox = []
            for k in self.extendbox_list:
                # if self.true_bbox_list[i][-18:-8] in k:
                if self.output_recover_list[i][-30:-20] in k:
                    extendbox = np.load(k)
                    print('extendbox: ', extendbox)
                    break

            for j in self.mhd_path_list:
                # if self.true_bbox_list[i][-18:-8] in j:
                if self.output_recover_list[i][-30:-20] in j:
                    bbox_temp_path = j
                    skip_append = False

            if bbox_temp_path is []:
                print("%s output list" % self.output_recover_list[i][-18:-8])
                skip_append = True
            if not skip_append:
                print(bbox_temp, bbox_temp_path, extendbox.shape)
                if bbox_temp.shape[0] > 0:
                    bbox_temp = out_coord_map(bbox_temp, bbox_temp_path, extendbox=extendbox)
                    if not (bbox_temp is []):
                        for j in range(bbox_temp.shape[0]):
                            if bbox_temp[j, 0] > 0:
                                self.true_bbox_result['seriesuid'].append(self.true_bbox_list[i][-18:-8])
                                self.true_bbox_result['coordX'].append(bbox_temp[j, 1])
                                self.true_bbox_result['coordY'].append(bbox_temp[j, 2])
                                self.true_bbox_result['coordZ'].append(bbox_temp[j, 3])
                                # self.true_bbox_result['diameter_mm'].append(bbox_temp[j, 4])
                                self.true_bbox_result['probability'].append(bbox_temp[j, 0])
                else:
                    print(bbox_temp_path, ": bbox temp path")  # 1181 1206 1155
            num_bbox -= 1
            print("num bbox", num_bbox)

        if self.true_bbox_result['seriesuid'].__len__() > 0:
            df = pd.DataFrame(self.true_bbox_result)
            df_id = df.seriesuid  # sort to right
            df = df.drop('seriesuid', axis=1)
            df.insert(0, 'seriesuid', df_id)
            T = time.localtime(time.time())
            df.to_csv(os.path.join(self.save_csv_dir, str(T.tm_mon) + '_' + str(T.tm_mday) + '_' + str(T.tm_hour) + '_' +
                                   str(T.tm_min) + '_conf' + str(conf_th) + '_' + 'detect' + str(nms_th) + '_result_' + self.process + '.csv'),
                                   index=False)

    def cut_nodules(self):
        pass


if __name__ == '__main__':
    if config['save_csv']:
        conf_th = [-1.5, -1, -0.8, 0, 1, 2]
        nms_th = [0.3, 0.5, 0.7]
        # detect_th = [0.1, 0.2, 0.3]
        detect_th = [0.1, 0.2, 0.3, 0.5, 0.7]
        # conf_th=0, nms_th=0.1, detect_th=0.1
        conf_th_temp = 7
        detect_th_temp = detect_th[0]
        nms_th_temp = 0.1

        save_csv = SaveCsv(config, process='test')
        save_csv(conf_th_temp, nms_th_temp, detect_th_temp)