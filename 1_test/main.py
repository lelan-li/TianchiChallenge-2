import os
import sys
os.environ['PYTHONPATH'] = '%s:%s' % ('/home/caffe/python', '/workspace/pai')
import sys
sys.path.append('/home/caffe/python')
sys.path.append('/workspace/pai')
import caffe

from data import DataBowl3Detector
from test_detect import test_detect
from split_combine import SplitComb
from test_config import test_config as config

process = 'test'
if config['detector']:
    net = caffe.Net(config['test_prototxt'], config['caffe_model'], caffe.TEST)
    split_comber = SplitComb(config)
    dataset = DataBowl3Detector(config, process=process, split_comber=split_comber)
    test_detect(dataset, net, config=config, process=process)
