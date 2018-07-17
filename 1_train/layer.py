import numpy as np
import random
import math
import os
os.environ['PYTHONPATH'] = '%s:%s' % ('/home/caffe/python', '/workspace/pai')
import sys
sys.path.append('/home/caffe/python')
sys.path.append('/workspace/pai')
import caffe

import data
from train_config import train_config as config


def sigmoid(x):
    return 1./(1 + math.e ** (-x))


def bce_loss(data, lable):
    data = sigmoid(data)
    if data <= 0:
        data = 0.000000000001
    elif data >= 1:
        data = 0.999999999999
    return -(lable*math.log(data) + (1 - lable)*math.log(1 - data))


def smooth_l1_loss(data, label):
    if math.fabs(data - label) < 1:
        return 0.5*(data - label)**2
    else:
        return math.fabs(data - label) - 0.5


class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)  # params
        self.top_names = ['data', 'label', 'coord']
        self.batch_size = int(param['batch_size'])
        self.indices = int(param['indices'])  # max indices
        self.idx = 0
        self.end_num = self.indices - self.batch_size
        self.input_size_data = tuple(param['data_size'])
        self.input_size_label = tuple(param['label_size'])
        self.input_size_coord = tuple(param['coord_size'])
        self.dataset_list = random.sample(range(self.indices), self.indices)  # random

        self.dataset_train = data.DataBowl3Detector(config, process='all')  # dataset
        # check init data shape
        top[0].reshape(self.batch_size, self.input_size_data[0], self.input_size_data[1],
                       self.input_size_data[2], self.input_size_data[3])
        top[1].reshape(self.batch_size, self.input_size_label[0], self.input_size_label[1],
                       self.input_size_label[2], self.input_size_label[3], self.input_size_label[4])
        top[2].reshape(self.batch_size, self.input_size_coord[0], self.input_size_coord[1],
                       self.input_size_coord[2], self.input_size_coord[3])
        print('DataLayer has done')

    def forward(self, bottom, top):
        for i in range(self.batch_size):  # batch by batch
            try:
                data, label, coord = self.dataset_train[self.dataset_list[(self.idx + i)]]
            except:
                print('bad image0: ', self.idx + i)
                select_change = self.idx + i - random.sample(range(10, 100), 1)[0]
                if select_change < 0:
                    select_change = -select_change
                if select_change >= self.indices:
                    select_change -= 110
                data, label, coord = self.dataset_train[self.dataset_list[select_change]]  # prevent data import failure

            if data.shape == self.input_size_data:
                top[0].data[i, ...] = data
                top[1].data[i, ...] = label
                top[2].data[i, ...] = coord
            else:
                print('bad image1: ', self.idx + i)
                select_change = self.idx + i - random.sample(range(10, 300), 1)[0]
                if select_change < 0:
                    select_change = -select_change
                if select_change >= self.indices:
                    select_change -= 310
                top[0].data[i, ...], \
                top[1].data[i, ...], \
                top[2].data[i, ...] = self.dataset_train[self.dataset_list[select_change]]  # prevent wrong shape
        self.idx += self.batch_size
        if self.idx > self.end_num:
            self.idx = 0
            self.dataset_list = random.sample(range(self.indices), self.indices)  # idx init

    def reshape(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass


class UnetLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.bath_count = -1
        self.show_rate = 15  # rate of print output
        self.show_loss_rate = 1  # rate of print loss
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        print('UnetLossLayer has done')

    def reshape(self, bottom, top):  # init
        if bottom[0].count != bottom[1].count:  # check input dimensions match
            print(bottom[0].data.shape)
            print(bottom[1].data.shape)
            raise Exception("Inputs must have the same dimension.")
        top[0].reshape(5)

    def forward(self, bottom, top):
        self.bath_count += 1
        self.batch_size = bottom[0].data.shape[0]  # data
        self.neg = []
        temp = np.asarray(bottom[1].data[:, :, 0, ...], dtype=np.float32)  # prevent caffe， python incompatible data
        self.pos = np.where(temp >= 0.5)  # positive sample
        neg_coord_old = np.asarray(np.where(temp <= -0.5))  # negative sample
        neg_num = min(len(neg_coord_old[0]), 6 * self.batch_size)
        if neg_num > 0:
            for i in random.sample(range(len(neg_coord_old[0])), neg_num):
                self.neg.append(neg_coord_old[:, i])

        top[0].data[:] = 0
        neg_loss = 0
        try:
            if len(self.neg) > 0:
                for i in range(len(self.neg)):
                    top[0].data[0] += bce_loss(bottom[0].data[self.neg[i][0], self.neg[i][1], 0,
                                                              self.neg[i][2], self.neg[i][3], self.neg[i][4]],
                                               bottom[1].data[self.neg[i][0], self.neg[i][1], 0,
                                                              self.neg[i][2], self.neg[i][3], self.neg[i][4]] + 1)
                    if self.bath_count % self.show_rate == 0:
                        print('neg_output: ', bottom[0].data[self.neg[i][0], self.neg[i][1], 0,
                                                             self.neg[i][2], self.neg[i][3], self.neg[i][4]])
                        # print('neg_labels: ', bottom[1].data[self.neg[i][0], self.neg[i][1], 0,
                        #                                      self.neg[i][2], self.neg[i][3], self.neg[i][4]])
                neg_loss = top[0].data[0]
            if len(self.pos[0]) > 0:
                for i in range(len(self.pos[0])):
                    top[0].data[0] += bce_loss(bottom[0].data[self.pos[0][i], self.pos[1][i], 0,
                                                              self.pos[2][i], self.pos[3][i], self.pos[4][i]],
                                               bottom[1].data[self.pos[0][i], self.pos[1][i], 0,
                                                              self.pos[2][i], self.pos[3][i], self.pos[4][i]])
                    if self.bath_count % self.show_rate == 0:
                        print('pos_output: ', bottom[0].data[self.pos[0][i], self.pos[1][i], 0, self.pos[2][i],
                                                             self.pos[3][i], self.pos[4][i]])
                        # print('pos_labels: ', bottom[1].data[self.pos[0][i], self.pos[1][i], 0, self.pos[2][i],
                        #                                      self.pos[3][i], self.pos[4][i]])
                    for m in range(1, 5):
                        top[0].data[m] += smooth_l1_loss(bottom[0].data[self.pos[0][i], self.pos[1][i], m,
                                                                        self.pos[2][i], self.pos[3][i], self.pos[4][i]],
                                                         bottom[1].data[self.pos[0][i], self.pos[1][i], m,
                                                                        self.pos[2][i], self.pos[3][i], self.pos[4][i]])
                        if self.bath_count % self.show_rate == 0:
                            print('pos_output_coord: ', m, bottom[0].data[self.pos[0][i], self.pos[1][i], m, self.pos[2][i],
                                                                          self.pos[3][i], self.pos[4][i]])
                            print('pos_labels_coord: ', m, bottom[1].data[self.pos[0][i], self.pos[1][i], m, self.pos[2][i],
                                                                          self.pos[3][i], self.pos[4][i]])
        except Exception as e:
            self.pos = [[]]
            self.neg = []
            print(Exception, ':', e)
            print('Error forward!')
        if self.bath_count % self.show_loss_rate == 0:
            print('--batch_count--%s--' % str(self.bath_count), '...classifier_loss: %s' % str(top[0].data[0]),
                  '...X_loss: %s' % str(top[0].data[1]*100/self.batch_size), 'Y_loss: %s' % str(top[0].data[2]*100/self.batch_size),
                  'Z_loss: %s' % str(top[0].data[3]*100/self.batch_size), 'R_loss: %s' % str(top[0].data[4]*100/self.batch_size))
            print('poss_loss: %s, neg_loss: %s' % ( str(top[0].data[0] - neg_loss), str(neg_loss)))

    def backward(self, top, propagate_down, bottom):  # bottom[0]: data bottom[1]: label
        bottom[0].diff[...] = np.zeros_like(bottom[0].data, dtype=np.float32)  # loss init
        try:
            if len(self.neg) > 0:
                for i in range(len(self.neg)):
                    neg_class_loss = sigmoid(bottom[0].data[self.neg[i][0], self.neg[i][1], 0, self.neg[i][2], self.neg[i][3], self.neg[i][4]]) \
                                     - bottom[1].data[self.neg[i][0], self.neg[i][1], 0, self.neg[i][2], self.neg[i][3], self.neg[i][4]] - 1
                    bottom[0].diff[self.neg[i][0], self.neg[i][1], 0, self.neg[i][2], self.neg[i][3], self.neg[i][4]] = neg_class_loss * 0.5
                    # if self.bath_count % self.show_rate == 0:
                    #     print('neg_diff: ', neg_class_loss*0.5)

            if len(self.pos[0]) > 0:
                for i in range(len(self.pos[0])):
                    pos_class_loss = sigmoid(bottom[0].data[self.pos[0][i], self.pos[1][i], 0, self.pos[2][i], self.pos[3][i], self.pos[4][i]])\
                                    - bottom[1].data[self.pos[0][i], self.pos[1][i], 0, self.pos[2][i], self.pos[3][i], self.pos[4][i]]
                    bottom[0].diff[self.pos[0][i], self.pos[1][i], 0, self.pos[2][i], self.pos[3][i], self.pos[4][i]] \
                        = pos_class_loss * 0.5
                    if self.bath_count % self.show_rate == 0:
                        print('pos_diff_class: ', pos_class_loss*0.5)
                    for n in range(1, 5):  # x y z r loss
                        x_sub_y = bottom[0].data[self.pos[0][i], self.pos[1][i], n, self.pos[2][i], self.pos[3][i], self.pos[4][i]]\
                                - bottom[1].data[self.pos[0][i], self.pos[1][i], n, self.pos[2][i], self.pos[3][i], self.pos[4][i]]
                        # print('x_sub_y:', n, x_sub_y)
                        if np.fabs(x_sub_y) <= 1:
                            bottom[0].diff[self.pos[0][i], self.pos[1][i], n, self.pos[2][i], self.pos[3][i], self.pos[4][i]] = x_sub_y
                            if self.bath_count % self.show_rate == 0:
                                print('pos_diff:', n, x_sub_y)
                        else:
                            bottom[0].diff[self.pos[0][i], self.pos[1][i], n, self.pos[2][i], self.pos[3][i], self.pos[4][i]] = np.sign(x_sub_y)
                            if self.bath_count % self.show_rate == 0:
                                print('pos_diff:', n, np.sign(x_sub_y))
        except Exception as e:
            print(Exception, ':', e)
            print('Error backward!')
        print('\n')
