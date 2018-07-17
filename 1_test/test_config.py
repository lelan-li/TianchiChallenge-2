code_folder = '/workspace/pai/ali_challenge2/'
data_folder = '/workspace/data/'
test_config = {'caffe_model': code_folder + '1_train/caffe_model/8_iter_150.caffemodel',
               'test_prototxt': code_folder + '1_test/test_0.prototxt',

               '1_test_csv_test': code_folder + '1_test/1_output_test/',
               '1_test_csv_val': code_folder + '1_test/1_output_val/',
               '1_test_csv_train': code_folder + '1_test/1_output_train/',

               'data_path_test': code_folder + 'prepare/prepare_result_test/',
               'data_path_val': code_folder + 'prepare/prepare_result_val/',
               'data_path_train': code_folder + 'prepare/prepare_result_train/',

               'true_bbox_folder_test': code_folder + '1_test/1_output_test/',
               'true_bbox_folder_val': code_folder + '1_test/1_output_val/',
               'true_bbox_folder_train': code_folder + '1_test/1_output_train/',

               'preprocess_result_test': code_folder + 'prepare/prepare_result_test/',
               'preprocess_result_val': code_folder + 'prepare/prepare_result_val/',
               'preprocess_result_train': code_folder + 'prepare/prepare_result_train/',

               'data_folder_train': data_folder + 'LKDS_MHD_TRAIN/LKDS_MHD_TRAIN/',
               'data_folder_val': data_folder + 'LKDS_MHD_VAL/LKDS_MHD_VAL/',
               'data_folder_test': data_folder + 'LKDS_MHD_TEST/LKDS_MHD_TEST/',

               'save_csv_dir': code_folder + '1_test/1_csv/',


               'detector': True,
               'save_csv': True,

               'anchors': [10, 30, 60],
               'chanel': 1,
               'crop_size': [128, 128, 128],
               'stride': 4,
               'max_stride': 16,
               'pad_value': 190,

               'sizelim': 2.,  # mm
               'sizelim2': 15,
               'sizelim3': 30,
               'reso': 1,

               'margin': 32,
               'side_len': 144,

               'blacklist': [],
               'bound_size': 12,
               'num_neg': 800,
               'th_neg': 0.02,
               'r_rand_crop': 0.3,
               'aug_scale': True,
               'augtype': {'flip': True,
                           'swap': False,
                           'scale': True,
                           'rotate': False}

               }