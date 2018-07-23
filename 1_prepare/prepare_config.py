data_folder = 'D:/data'
code_folder = 'D:/ali_challenge2'
prepare_config = {'data_folder': data_folder,

                  'data_folder_train': data_folder + '/LKDS_MHD_TRAIN/LKDS_MHD_TRAIN',
                  'data_folder_val': data_folder + '/LKDS_MHD_VAL/LKDS_MHD_VAL',
                  'data_folder_test': data_folder + '/LKDS_MHD_TEST/LKDS_MHD_TEST',
                  'data_folder_test2': data_folder + '/test2',
                  'annotations_path_train': data_folder + '/csv/train/annotations.csv',
                  'annotations_path_val': data_folder + '/csv/val/annotations.csv',

                  'code_folder': code_folder,

                  'prepare_result_train': './prepare_result_train',
                  'prepare_result_val': './prepare_result_val',
                  'prepare_result_test': './prepare_result_test',
                  'prepare_result_test2': './prepare_result_test2',

                  'prepare_check_train': './prepare_check_train',
                  'prepare_check_val': './prepare_check_val',
                  'prepare_check_test': './prepare_check_test',
                  'prepare_check_test2': './prepare_check_test2',

                  'n_worker_preprocessing': 16,
                  'use_exsiting_preprocessing': False}