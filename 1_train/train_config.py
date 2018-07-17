code_folder = '/workspace/pai/ali_challenge2/'
train_config = {'anchors': [10.0, 25.0, 35.],
                'chanel': 1,
                'crop_size': [128, 128, 128],
                'stride': 4,
                'max_stride': 16,
                'num_neg': 800,
                'th_neg': 0.02,
                'th_pos_train': 0.5,
                'th_pos_val': 1,
                'num_hard': 2,
                'bound_size': 12,
                'reso': 1,

                'sizelim': 2.,
                'sizelim2': 15,
                'sizelim3': 30,

                'aug_scale': True,
                'r_rand_crop': 0.3,
                'pad_value': 190,
                'augtype': {'flip': True,
                            'swap': False,
                            'scale': True,
                            'rotate': False},

                'argsbatch_size': 16,
                'argsworkers': 32,
                'preprocess_result_train': code_folder + 'prepare/prepare_result_train',
                'preprocess_result_val': code_folder + 'prepare/prepare_result_val',
                'preprocess_result_test': code_folder + 'prepare/prepare_result_test',
                'preprocess_result_all': code_folder + 'prepare/prepare_result_all',

                'annotations_path_train': code_folder + '/csv/train/annotations.csv',
                'annotations_path_val': code_folder + '/csv/val/annotations.csv',

                'code_folder': '/home/tianchi/',

                'prepare_result_train': './prepare_result_train',
                'prepare_result_val': './prepare_result_val',
                'prepare_result_test': './prepare_result_test2',
                'prepare_result_test2': './prepare_result_test2',
                'prepare_result_all': './prepare_result_all',

                'blacklist': ['LKDS-01836', 'LKDS-00906'],  # un-process

                'margin': 32,
                'sidelen': 144
                }
