import numpy as np
import os
from test_func import GetPBB


def test_detect(dataset, net, config, process):
    split_comber = dataset.split_comber
    num = len(dataset)

    for i in range(len(dataset)):
        data, coord, nzhw = dataset[i]
        outputlist = np.zeros([data.shape[0], 52, 52, 52, 3, 5])
        shortname = dataset.all_filename_list()[i]
        for j in range(data.shape[0]):
            net.blobs['data'].data[0, ...] = data[j, ...]
            net.blobs['coord'].data[0, ...] = coord[j, ...]
            output_temp = net.forward()
            output = output_temp['out2']
            output = np.asarray(output, dtype=np.float32)
            output = np.transpose(output, [0, 3, 4, 5, 1, 2])
            outputlist[j, ...] = output[0, ...]

        output = split_comber.combine(outputlist, nzhw=nzhw)
        thresh = -3
        getpbb = GetPBB(config)
        pbb, where, output_recover, output_raw = getpbb(output, thresh, ismask=True)  # numpy.ndarray: (239, 5)

        print([i, shortname])
        np.save(os.path.join(config['1_test_csv_' + process], shortname + '_output_raw.npy'), output_raw)
        np.save(os.path.join(config['1_test_csv_' + process], shortname + '_output_recover.npy'), output_recover)
        # np.save(os.path.join(config['1_test_csv_' + process], shortname + '_where.npy'), where)
        np.save(os.path.join(config['1_test_csv_' + process], shortname+'_pbb.npy'), pbb)
        num = (num - 1)
        print('ʣ��:', num)