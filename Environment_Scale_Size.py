import matplotlib.pyplot as plt
from data_process import DataProcess
import os

import numpy as np
import random

r = random.random
random.seed(1)


A_0_bound = 35.0
A_1_bound = 30.0
A_1_bias = 45.0

Color_List = ['r', 'g', 'b', 'y', 'w']

train_img_list = os.listdir('train_data/images')
test_img_list = os.listdir('test_data/images')

train_img_num = len(train_img_list)
test_img_num = len(test_img_list)


print('train img num', train_img_num)
print('train img list', train_img_list)

print('test img num', test_img_num)


print("\033[1;34;34mInput\033[0m")
print("\033[4;31;31mInput\033[0m")


class Environment(object):
    def __init__(self, is_train=True):

        self.reward = 0
        self.stop_step = 17
        self.step_count = 0

        self.start_point = [125, 0]

        self.img_idx = 0

        self.plt_show = False

        self.data_set = 'Train set'

        if is_train:
            self.is_train = True
            self.img_num = train_img_num
            self.data_process = DataProcess(
                label_path='train_data/labels/',
                img_path='train_data/images/',
                resized_W=250,
                resized_H=750,
                patch_size=100,
                mask_size=50)
        else:
            self.is_train = False
            self.img_num = test_img_num
            print('\n * * * * * * * * * * * * Test Set * * * * * * * * * * * * * *')
            self.data_process = DataProcess(
                label_path='test_data/labels/',
                img_path='test_data/images/',
                resized_W=250,
                resized_H=750,
                patch_size=100,
                mask_size=50)

    def reset(self, img_idx=0):

        """

        :param img_idx:     the img_idx
        :return:            start_point:     p_0
                            img_patch_0:     patch_0
        """

        if self.is_train:
            self.data_set = 'Train Set'
            self.img_idx = img_idx
        else:
            self.data_set = 'Test Set'
            self.img_idx = test_img_list[img_idx]

        print('\nEnv reset !   ' + self.data_set + '   Img ID: {0}'.format(self.img_idx))

        start_point = self.start_point

        state_patch = self.data_process.make_state(self.img_idx, start_point)

        return start_point, state_patch

    def step(self, a, p):

        """
        s_, r, done = env.step(a)       p --- a ---> p_
        :param a:
        :return:
        """

        action = a[0]

        action_x = action[0] * A_0_bound
        action_y = action[1] * A_1_bound + A_1_bias

        p_ = np.zeros((2,), dtype='float32')

        p_[0] = p[0] + action_x
        p_[1] = p[1] + action_y

        if p_[0] <= 0:
            p_[0] = 0.0
        elif p_[0] > 245.0:
            p_[0] = 245.0

        if p_[1] > 745:
            p_[1] = 745.0

        state_ = self.data_process.make_state(self.img_idx, p_)

        return p_, state_

    def make_s_input(self, s):

        """
        process the state data information

        :param  s:      the output of make_state()   (1, 400, 400, 2)    uint8

        :return:        s_input : data input to the network     (400, 400, 2)     float32
        """

        s_scale = s.astype('float32') / 255.0   # (1, 400, 400, 2)  float32

        s_input = s_scale[0, :, :, :]           # (400, 400, 2)     float32

        return s_input

    def get_all_17_label(self):

        all_17_points_17 = self.data_process.make_point(self.img_idx)

        return all_17_points_17

    def get_label_i(self, vb_idx):

        '''
        get label center point of the vb_idx th VB of current img
        :param vb_idx:      the VB index

        :return:            Label: center label of the vb_idx th VB : point_i
        '''

        all_17_points = self.data_process.make_point(self.img_idx)

        point_i = all_17_points[vb_idx]

        plt_img = False

        if plt_img:

            fig3 = plt.figure(3)

            _, img = self.data_process.load_img(self.img_idx)

            plt.imshow(img)

            # plt all 17 VB center points
            for vb_idx in range(17):

                plt.plot(all_17_points[vb_idx, 0], all_17_points[vb_idx, 1], 'ro-', markersize=2)

            # plt current VB points
            plt.plot(point_i[0], point_i[1], 'bo-', markersize=4)

            print('\npoint_i: ', point_i)

        return point_i

    def get_landmarks_i(self, vb_idx):

        landmarks = self.data_process.make_label_scaled(self.img_idx)  # 68 points  range: 0-1     [68, 2]
        landmark_i = landmarks[4 * vb_idx: 4*(vb_idx+1), :]     # 4 points      range : 0-1     [4, 2]

        landmark_i[:, 0] = landmark_i[:, 0] * 250
        landmark_i[:, 1] = landmark_i[:, 1] * 750

        return landmark_i

    def get_all_landmarks(self):

        landmarks = self.data_process.make_label_scaled(self.img_idx)  # 68 points  range: 0-1     [68, 2]

        landmarks[:, 0] = landmarks[:, 0] * 250
        landmarks[:, 1] = landmarks[:, 1] * 750

        return landmarks

    def get_patch_i(self, vb_idx):
        '''
        get label img patch
        :param vb_idx:
        :return:
        '''

        point_i = self.get_label_i(vb_idx)

        img_patch_i = self.data_process.make_patch_with_point(self.img_idx, point_i)

        fig4 = plt.figure(4)

        plt.imshow(img_patch_i[0, :, :, :])

    def cal_reward(self, p_gt, p_pre):

        reward = p_gt - p_pre

        return reward



