import scipy.io as scio
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

color_list = ['b', 'g', 'r', 'y', 'w']
Color_List = ['r', 'g', 'b', 'y', 'c', 'm', 'w']


def data_split(full_list, ratio, random_seed, shuffle=False):
    """

    """
    total_num = len(full_list)
    offset = int(total_num * ratio)
    if total_num == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list, random=random_seed)
    sublist_1 = full_list[: offset]
    sublist_2 = full_list[offset:]

    return sublist_1, sublist_2


class DataProcess(object):

    def __init__(self,
                 label_path='dataset/train_label/',
                 img_path='dataset/train_img/',
                 resized_W=500,
                 resized_H=1500,
                 patch_size=200,
                 mask_size=100,
                 att_size=50):
        self.label_path = label_path
        self.img_path = img_path
        self.resized_W = resized_W
        self.resized_H = resized_H
        self.patch_size = patch_size

        self.mask_size = mask_size

        self.att_size = att_size

    global img_path, img_dirs
    img_path = "dataset/train_img/"
    img_dirs = os.listdir(img_path)

    def load_img(self, index, add_noise=False):
        """

        :param   index:     int     index of img
        :return: img:               origin img form data set (H, W, 3)
                 img_resize         resized img with size    (resized_H, resized_W, 3)
        """

        # img_dirs = os.listdir(self.img_path)
        #
        # print("===img_path===:", self.img_path)

        # img_path_name = os.path.join(self.img_path, img_dirs[index])
        img_path_name = os.path.join(img_path, img_dirs[index])

        img = cv2.imread(img_path_name)

        b, g, r = cv2.split(img)

        img = cv2.merge([r, g, b])

        img_resize = cv2.resize(img, (self.resized_W, self.resized_H))

        if add_noise:

            # add Gaussian Noise
            image = np.array(img_resize / 255, dtype=float)

            mean = 0
            var = 0.00008
            noise = np.random.normal(mean, var ** 0.5, image.shape)

            out = image + noise

            if out.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.

            out = np.clip(out, low_clip, 1.0)
            out = np.uint8(out * 255)
            img_resize = out
            # add Gaussian Noise

        return img, img_resize

    def load_att_img(self, index, att_img_dir='entire_dataset/train_img_with_loc/'):
        """

        """
        img_dirs = os.listdir(self.img_path)

        # att_img_dir = 'entire_dataset/train_img_with_loc/'
        att_img_path = os.path.join(att_img_dir, img_dirs[index])

        img = cv2.imread(att_img_path)

        print('Load att img : [{0}]   img shape : {1}'.format(att_img_path, img.shape))

        b, g, r = cv2.split(img)

        img = cv2.merge([r, g, b])

        img_resize = cv2.resize(img, (self.resized_W, self.resized_H))

        return img, img_resize

    def dis_img_with_att_region(self, index, points_dir='Loc_Result/train_set/Pre_Points/'):

        _, img = self.load_img(index)

        att_mask_img = np.zeros(shape=img.shape, dtype=np.uint8)
        pre_center_p = np.load(points_dir + str(index) + '.npy')
        for vb_idx in range(17):

            w_from = int(pre_center_p[vb_idx, 0]) - self.mask_size
            w_to = int(pre_center_p[vb_idx, 0]) + self.mask_size
            if w_from < 0:
                w_from = 0
            if w_to > self.resized_W:
                w_to = self.resized_W

            h_from = int(pre_center_p[vb_idx, 1]) - self.mask_size
            h_to = int(pre_center_p[vb_idx, 1]) + self.mask_size
            if h_from < 0:
                h_from = 0
            if h_to > self.resized_H:
                h_to = self.resized_H

            att_mask_img[h_from:h_to, w_from:w_to, :] = 1

        img_with_att = img * att_mask_img

        show_img = False
        if show_img:

            f1 = plt.figure(index)
            plt.title('img with att')
            plt.plot(pre_center_p[:, 0], pre_center_p[:, 1], 'rx', markersize=4)
            plt.imshow(img_with_att)
            plt.show()

        return img_with_att

    def save_img_with_att(self, index, points_dir='Loc_Result/train_set/Pre_Points/'):
        """
        Train Set
        """
        img_dirs = os.listdir(self.img_path)

        att_img_save_dir = 'dataset/img_with_att/'
        att_img_save_path = os.path.join(att_img_save_dir, img_dirs[index])

        origin_img, img = self.load_img(index)
        width = origin_img.shape[1]
        height = origin_img.shape[0]

        att_mask_img = np.zeros(shape=img.shape, dtype=np.uint8)
        pre_center_p = np.load(points_dir + str(index) + '.npy')

        for vb_idx in range(17):

            if vb_idx > 12:
                att_size = 60
            else:
                att_size = 50

            w_from = int(pre_center_p[vb_idx, 0]) - att_size
            w_to = int(pre_center_p[vb_idx, 0]) + att_size
            if w_from < 0:
                w_from = 0
            if w_to > self.resized_W:
                w_to = self.resized_W

            h_from = int(pre_center_p[vb_idx, 1]) - self.mask_size
            h_to = int(pre_center_p[vb_idx, 1]) + self.mask_size
            if h_from < 0:
                h_from = 0
            if h_to > self.resized_H:
                h_to = self.resized_H

            att_mask_img[h_from:h_to, w_from:w_to, :] = 1

        img_with_att = img * att_mask_img

        img_with_att = cv2.resize(img_with_att, (width, height))

        save_img = True
        if save_img:
            cv2.imwrite(att_img_save_path, img_with_att)
            print('save into img_path_name : ', att_img_save_path)

        show_img = False
        if show_img:
            f1 = plt.figure(index)
            plt.title('img with att')
            plt.imshow(img_with_att)
            plt.show()

        return img_with_att

    def save_img_with_att_test_set(self, index, points_dir='Loc_Result/test_set/'):
        """
        Test Set
        """
        img_dirs = os.listdir(self.img_path)

        att_img_save_dir = 'test_dataset/test_img_with_loc/'
        att_img_save_path = os.path.join(att_img_save_dir, img_dirs[index])

        origin_img, img = self.load_img(index)
        width = origin_img.shape[1]
        height = origin_img.shape[0]

        att_mask_img = np.zeros(shape=img.shape, dtype=np.uint8)
        pre_center_p = np.load(points_dir + str(index) + '.npy')

        for vb_idx in range(17):

            w_from = int(pre_center_p[vb_idx, 0]) - self.mask_size
            w_to = int(pre_center_p[vb_idx, 0]) + self.mask_size
            if w_from < 0:
                w_from = 0
            if w_to > self.resized_W:
                w_to = self.resized_W

            h_from = int(pre_center_p[vb_idx, 1]) - self.mask_size
            h_to = int(pre_center_p[vb_idx, 1]) + self.mask_size
            if h_from < 0:
                h_from = 0
            if h_to > self.resized_H:
                h_to = self.resized_H

            att_mask_img[h_from:h_to, w_from:w_to, :] = 1

        img_with_att = img * att_mask_img

        img_with_att = cv2.resize(img_with_att, (width, height))

        save_img = True
        if save_img:
            cv2.imwrite(att_img_save_path, img_with_att)
            print('save into img_path_name : ', att_img_save_path)

        show_img = False
        if show_img:
            f1 = plt.figure(index)
            plt.title('img with att')
            plt.imshow(img_with_att)
            plt.show()

        return img_with_att

    def load_label_old(self, index):
        """

        :param      index:      index of the label for data set
        :return:    label:      label (uint16, (68, 2))
        """

        label_dirs = os.listdir(self.label_path)

        label_path_name = os.path.join(self.label_path, label_dirs[index])

        label_mat = scio.loadmat(label_path_name)

        label = label_mat['p2']

        return label

    def load_label(self, index):
        """

        :param      index:      index of the label for data set
        :return:    label:      label (uint16, (68, 2))
        """

        # img_dirs = os.listdir(self.img_path)

        label_path_name = os.path.join(self.label_path, img_dirs[index] + ".mat")

        label_mat = scio.loadmat(label_path_name)

        label = label_mat['p2']

        return label

    def make_label_scaled(self, index):
        """

        :param      index:          int                 index of img/label
        :return:    label_scaled    ndarray (68, 2)     scaled label  (range:[0, 1])
        """

        img, _ = self.load_img(index)

        label = self.load_label(index)

        img_w, img_h = img.shape[1], img.shape[0]

        # 每块VB 68个点(17块VB * 每块VB 4 个点)
        label_scaled = np.ndarray((68, 2), dtype='float32')
        label_scaled[:, 0] = label[:, 0] / img_w
        label_scaled[:, 1] = label[:, 1] / img_h

        return label_scaled

    def make_point(self, index):

        """
        :param      index:      int             index of the img
        :return:    center_p    ndarray(17, 2)  center points of 17 VBs
        """
        label_scaled = self.make_label_scaled(index)

        label_new = np.ndarray((68, 2), dtype='float32')
        label_new[:, 0] = label_scaled[:, 0] * self.resized_W
        label_new[:, 1] = label_scaled[:, 1] * self.resized_H

        mid_p = np.ndarray((34, 2), dtype='float32')

        for i in range(17):
            mid_p[2 * i, :] = (label_new[4 * i, :] + label_new[4 * i + 2, :]) / 2
            mid_p[2 * i + 1, :] = (label_new[4 * i + 1, :] + label_new[4 * i + 3, :]) / 2

        center_p = np.ndarray((17, 2), dtype='float32')

        for i in range(17):
            center_p[i, :] = (mid_p[2 * i, :] + mid_p[2 * i + 1, :]) / 2

        return center_p

    def get_vb_size(self, index, vb_idx):
        """

        :param index:       image index
        :param vb_idx:      vb index
        :return:            vb size
        """

        # center_point = self.make_point(index)

        label_scaled = self.make_label_scaled(index)

        label_new = np.ndarray((68, 2), dtype='float32')
        label_new[:, 0] = label_scaled[:, 0] * self.resized_W
        label_new[:, 1] = label_scaled[:, 1] * self.resized_H

        # vb_1 size
        y_max = max(label_new[4 * vb_idx, 1], label_new[4 * vb_idx + 1, 1],
                    label_new[4 * vb_idx + 2, 1], label_new[4 * vb_idx + 3, 1])
        y_min = min(label_new[4 * vb_idx, 1], label_new[4 * vb_idx + 1, 1],
                    label_new[4 * vb_idx + 2, 1], label_new[4 * vb_idx + 3, 1])

        x_max = max(label_new[4 * vb_idx, 0], label_new[4 * vb_idx + 1, 0],
                    label_new[4 * vb_idx + 2, 0], label_new[4 * vb_idx + 3, 0])
        x_min = min(label_new[4 * vb_idx, 0], label_new[4 * vb_idx + 1, 0],
                    label_new[4 * vb_idx + 2, 0], label_new[4 * vb_idx + 3, 0])

        h = y_max - y_min
        w = x_max - x_min

        vb_size = max(h, w)

        return vb_size

    def img_display(self, index):

        save_dis = False

        dis_landmarks = False
        dis_center_p = True
        dis_vb_size = True

        plt.close()
        fig = plt.figure(index)

        _, img_resize = self.load_img(index)


        plt.imshow(img_resize)

        center_point = self.make_point(index)

        label_scaled = self.make_label_scaled(index)

        label_new = np.ndarray((68, 2), dtype='float32')
        label_new[:, 0] = label_scaled[:, 0] * self.resized_W
        label_new[:, 1] = label_scaled[:, 1] * self.resized_H

        # plot landmarks of each VB(17 * 4 = 68) with color yellow
        if dis_landmarks:
            for i in range(68):
                plt.plot(label_new[i, 0], label_new[i, 1], 'wo-', markersize=2)
                plt.text(label_new[i, 0], label_new[i, 1], ''+str(i + 1), size=7, alpha=0.9, color=color_list[i//4 % 5])

        # plot center point of each VB with color red
        if dis_center_p:
            for i in range(17):
                plt.plot(center_point[i, 0], center_point[i, 1], color_list[i % 5] + 'o-', markersize=2)
                plt.text(center_point[i, 0], center_point[i, 1], str(i + 1), size=7, alpha=0.9, color=color_list[i % 5])

        if dis_vb_size:
            for i in range(17):
                vb_size = self.get_vb_size(index, i)
                plt.text(center_point[i, 0], center_point[i, 1],
                         '         ' + str(vb_size), size=7, alpha=0.9, color=color_list[i % 5])
        if save_dis:
            plt.savefig('swh_dataset/display/img_' + str(index) + '.png', bbox_inches='tight')
        else:
            plt.show()

    def landmarks_display(self, index):

        save_dis = False

        dis_landmarks = True
        dis_center_p = False
        dis_vb_size = True

        plt.close()
        fig = plt.figure(index)

        _, img_resize = self.load_img(index)

        plt.imshow(img_resize)

        center_point = self.make_point(index)

        label_scaled = self.make_label_scaled(index)

        label_new = np.ndarray((68, 2), dtype='float32')
        label_new[:, 0] = label_scaled[:, 0] * self.resized_W
        label_new[:, 1] = label_scaled[:, 1] * self.resized_H

        # plot landmarks of each VB(17 * 4 = 68) with color yellow
        if dis_landmarks:
            for i in range(68):
                plt.plot(label_new[i, 0], label_new[i, 1], Color_List[i//4 % 7] + 'o-', markersize=1)
                plt.text(label_new[i, 0], label_new[i, 1], ''+str(i + 1), size=4, alpha=0.9, color=Color_List[i//4 % 7])

        # plot center point of each VB with color red
        if dis_center_p:
            for i in range(17):
                plt.plot(center_point[i, 0], center_point[i, 1], Color_List[i % 5] + 'o-', markersize=2)
                plt.text(center_point[i, 0], center_point[i, 1], str(i + 1), size=7, alpha=0.9, color=Color_List[i % 5])

        if dis_vb_size:
            for i in range(17):
                vb_size = self.get_vb_size(index, i)
                plt.text(center_point[i, 0], center_point[i, 1],
                         '         ' + str(vb_size), size=7, alpha=0.9, color=Color_List[i % 7])
        if save_dis:
            plt.savefig('swh_dataset/display/img_' + str(index) + '.png', bbox_inches='tight')
        else:
            plt.show()

    def att_img_display(self, index):

        fig = plt.figure(index)

        _, img_resize = self.load_att_img(index)

        plt.imshow(img_resize)

        center_point = self.make_point(index)

        label_scaled = self.make_label_scaled(index)

        label_new = np.ndarray((68, 2), dtype='float32')
        label_new[:, 0] = label_scaled[:, 0] * self.resized_W
        label_new[:, 1] = label_scaled[:, 1] * self.resized_H

        # plot center point of each VB with color red
        for i in range(17):
            plt.plot(center_point[i, 0], center_point[i, 1], 'ro-', markersize=2)
            plt.text(center_point[i, 0], center_point[i, 1], str(i + 1), size=10, alpha=0.9, color=color_list[i % 5])

        plt.show()

    def make_patch(self, index):
        """

        :param      index:
        :return:    img_patch   (17, 400, 400, 3)
        """

        img_patch = np.ndarray((17, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        center_p = self.make_point(index)

        _, img_resize = self.load_img(index)

        for i in range(0, 17):
            if (center_p[i, 1] - self.patch_size) <= 0:
                h_from, h_to = 0, 2 * self.patch_size
            elif (center_p[i, 1] + self.patch_size) > self.resized_H:
                h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H
            else:
                h_from, h_to = int(center_p[i, 1]) - self.patch_size, int(center_p[i, 1]) + self.patch_size

            #
            if (center_p[i, 0] - self.patch_size) <= 0:
                w_from, w_to = 0, 2 * self.patch_size
            elif (center_p[i, 0] + self.patch_size) > self.resized_W:
                w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W
            else:
                w_from, w_to = int(center_p[i, 0]) - self.patch_size, int(center_p[i, 0]) + self.patch_size

            img_patch[i, :, :, :] = img_resize[h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_patch_s0(self, index):

        img_patch_s0 = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)
        _, img_resize = self.load_img(index)
        img_patch_s0[0, :, :, :] = img_resize[0: 400, 250 - 200:250 + 200, :]

        return img_patch_s0

    def make_patch_with_point(self, index, point):

        img_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        _, img_resize = self.load_img(index)

        center_point = np.zeros((2,), dtype='float32')
        center_point[0] = point[0]
        center_point[1] = point[1]


        if (center_point[1] - self.patch_size) <= 0:
            h_from, h_to = 0, 2 * self.patch_size

        elif (center_point[1] + self.patch_size) >= self.resized_H:
            h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H

        else:
            h_from = int(center_point[1]) - self.patch_size
            h_to = int(center_point[1]) + self.patch_size

        if (center_point[0] - self.patch_size) <= 0:
            w_from, w_to = 0, 2 * self.patch_size

        elif (center_point[0] + self.patch_size) >= self.resized_W:
            w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W

        else:
            w_from = int(center_point[0]) - self.patch_size
            w_to = int(center_point[0]) + self.patch_size

        # cv2.imshow("img_resize:", img_resize)
        # cv2.waitKey(0)

        img_patch[0, :, :, :] = img_resize[h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_mask_patch_with_point(self, index, point):

        img_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        _, img_resize = self.load_img(index)

        center_point = point

        self.mask_size1 = self.mask_size + 10

        if (center_point[1] - self.mask_size1) <= 0:
            h_from, h_to = 0, 2 * self.mask_size1

        elif (center_point[1] + self.mask_size1) > self.resized_H:
            h_from, h_to = self.resized_H - 2 * self.mask_size1, self.resized_H

        else:
            h_from, h_to = int(center_point[1]) - self.mask_size1, int(center_point[1]) + self.mask_size1

        if (center_point[0] - self.mask_size1) <= 0:
            w_from, w_to = 0, 2 * self.mask_size1

        elif (center_point[0] + self.mask_size1) > self.resized_W:
            w_from, w_to = self.resized_W - 2 * self.mask_size1, self.resized_W

        else:
            w_from, w_to = int(center_point[0]) - self.mask_size1, int(center_point[0]) + self.mask_size1

        mask_img = np.zeros((1, 1500, 500, 3), dtype=np.uint8)
        mask_img[0, h_from: h_to, w_from:w_to, :] = img_resize[h_from: h_to, w_from:w_to, :]

        if (center_point[1] - self.patch_size) <= 0:
            h_from, h_to = 0, 2 * self.patch_size

        elif (center_point[1] + self.patch_size) > self.resized_H:
            h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H

        else:
            h_from, h_to = int(center_point[1]) - self.patch_size, int(center_point[1]) + self.patch_size

        if (center_point[0] - self.patch_size) <= 0:
            w_from, w_to = 0, 2 * self.patch_size

        elif (center_point[0] + self.patch_size) > self.resized_W:
            w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W

        else:
            w_from, w_to = int(center_point[0]) - self.patch_size, int(center_point[0]) + self.patch_size

        # cv2.imshow("mask_img:", mask_img[0, h_from: h_to, w_from:w_to, :])
        # cv2.waitKey(0)
        img_patch[0, :, :, :] = mask_img[0, h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_mask_patch_with_landmark(self, index, landmark, point):

        img_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        _, img_resize = self.load_img(index)

        x_max = max(landmark[0, 0], landmark[1, 0], landmark[2, 0], landmark[3, 0])
        x_min = min(landmark[0, 0], landmark[1, 0], landmark[2, 0], landmark[3, 0])

        y_max = max(landmark[0, 1], landmark[1, 1], landmark[2, 1], landmark[3, 1])
        y_min = min(landmark[0, 1], landmark[1, 1], landmark[2, 1], landmark[3, 1])

        extend_dis = 15
        x_min = int(x_min - extend_dis)
        x_max = int(x_max + extend_dis)

        y_min = int(y_min - extend_dis)
        y_max = int(y_max + extend_dis)

        if x_min <= 0:
            x_min = 0
        if x_max >= 2*self.resized_W:
            x_max = 2*self.resized_W
        if y_min <= 0:
            x_min = 0
        if y_max >= 2*self.resized_H:
            x_max = 2*self.resized_H

        h_from, h_to, w_from, w_to = y_min, y_max, x_min, x_max

        mask_img = np.zeros((1, self.resized_H, self.resized_W, 3), dtype=np.uint8)
        mask_img[0, h_from: h_to, w_from:w_to, :] = img_resize[h_from: h_to, w_from:w_to, :]

        center_point = point
        if (center_point[1] - self.patch_size) <= 0:

            h_from, h_to = 0, 2 * self.patch_size

        elif (center_point[1] + self.patch_size) > self.resized_H:

            h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H

        else:

            h_from, h_to = int(center_point[1]) - self.patch_size, int(center_point[1]) + self.patch_size

        if (center_point[0] - self.patch_size) <= 0:

            w_from, w_to = 0, 2 * self.patch_size

        elif (center_point[0] + self.patch_size) > self.resized_W:

            w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W

        else:

            w_from, w_to = int(center_point[0]) - self.patch_size, int(center_point[0]) + self.patch_size

        img_patch[0, :, :, :] = mask_img[0, h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_patch_with_point_v2(self, index, point):

        img_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        _, img_resize = self.load_img(index)

        center_point = point

        if (center_point[1] - self.patch_size / 2) <= 0:

            h_from, h_to = 0, 2 * self.patch_size

        elif (center_point[1] + self.patch_size * 3 / 2) > self.resized_H:

            h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H

        else:

            h_from = int(center_point[1]) - int(self.patch_size / 2)
            h_to = int(center_point[1]) + int(self.patch_size * 3 / 2)

        if (center_point[0] - self.patch_size) <= 0:

            w_from, w_to = 0, 2 * self.patch_size

        elif (center_point[0] + self.patch_size) > self.resized_W:

            w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W

        else:

            w_from, w_to = int(center_point[0]) - self.patch_size, int(center_point[0]) + self.patch_size

        img_patch[0, :, :, :] = img_resize[h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_mask_patch_with_point_v2(self, index, point):

        img_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 3), dtype=np.uint8)

        _, img_resize = self.load_img(index)

        center_point = point

        if (center_point[1] - self.patch_size / 2) <= 0:

            h_from, h_to = 0, 2 * self.mask_size

        elif (center_point[1] + self.patch_size / 2) > self.resized_H:

            h_from, h_to = self.resized_H - 2 * self.mask_size, self.resized_H

        else:

            h_from = int(center_point[1]) - int(self.patch_size / 2)
            h_to = int(center_point[1]) + int(self.patch_size / 2)

        if (center_point[0] - self.mask_size) <= 0:

            w_from, w_to = 0, 2 * self.mask_size

        elif (center_point[0] + self.mask_size) > self.resized_W:

            w_from, w_to = self.resized_W - 2 * self.mask_size, self.resized_W

        else:

            w_from, w_to = int(center_point[0]) - self.mask_size, int(center_point[0]) + self.mask_size

        mask_img = np.zeros((1, 750, 250, 3), dtype=np.uint8)
        mask_img[0, h_from: h_to, w_from:w_to, :] = img_resize[h_from: h_to, w_from:w_to, :]

        if (center_point[1] - self.patch_size / 2) <= 0:

            h_from, h_to = 0, 2 * self.patch_size

        elif (center_point[1] + self.patch_size * 3 / 2) > self.resized_H:

            h_from, h_to = self.resized_H - 2 * self.patch_size, self.resized_H

        else:

            h_from = int(center_point[1]) - int(self.patch_size / 2)
            h_to = int(center_point[1]) + int(self.patch_size * 3 / 2)

        if (center_point[0] - self.patch_size) <= 0:

            w_from, w_to = 0, 2 * self.patch_size

        elif (center_point[0] + self.patch_size) > self.resized_W:

            w_from, w_to = self.resized_W - 2 * self.patch_size, self.resized_W

        else:

            w_from, w_to = int(center_point[0]) - self.patch_size, int(center_point[0]) + self.patch_size

        img_patch[0, :, :, :] = mask_img[0, h_from: h_to, w_from:w_to, :]

        return img_patch

    def make_state(self, index, point):

        state_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 2), dtype=np.uint8)
        img_patch = self.make_patch_with_point(index, point)

        mask_patch = self.make_mask_patch_with_point(index, point)  # (1, 400, 400, 3) --- dtype : uint8

        state_patch[:, :, :, 0] = img_patch[:, :, :, 0]
        state_patch[:, :, :, 1] = mask_patch[:, :, :, 0]

        return state_patch

    def make_state_with_landmark(self, index, landmark):

        state_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 2), dtype=np.uint8)

        point = self.landmarks_2_point(landmark)
        img_patch = self.make_patch_with_point(index, point)
        mask_patch = self.make_mask_patch_with_landmark(index, landmark, point)

        state_patch[:, :, :, 0] = img_patch[:, :, :, 0]
        state_patch[:, :, :, 1] = mask_patch[:, :, :, 0]

        return state_patch

    def landmarks_2_point(self, landmarks):
        """
        landmarks   :   [4, 2]
        point       :   [1, 2]
        """

        # print(landmarks)
        l_point = (landmarks[0, :] + landmarks[2, :]) / 2
        r_point = (landmarks[1, :] + landmarks[3, :]) / 2

        c_point = (l_point + r_point) / 2

        return c_point

    def make_state_v2(self, index, point):

        state_patch = np.ndarray((1, 2 * self.patch_size, 2 * self.patch_size, 2), dtype=np.uint8)

        img_patch = self.make_patch_with_point_v2(index, point)     # (1, 200, 200, 3) --- dtype : uint8

        mask_patch = self.make_mask_patch_with_point_v2(index, point)   # (1, 400, 400, 3) --- dtype : uint8

        state_patch[:, :, :, 0] = img_patch[:, :, :, 0]
        state_patch[:, :, :, 1] = mask_patch[:, :, :, 0]

        return state_patch

    def state_display(self, index, vb_index):
        """
        Display the State patch
        :param index:
        :param vb_index:
        :return:
        """

        _, img_resize = self.load_img(index)

        f1 = plt.figure()
        plt.title('Image')
        plt.imshow(img_resize)

        center_point = self.make_point(index)

        plt.plot(center_point[vb_index, 0], center_point[vb_index, 1], 'ro-', markersize=2)

        plt.text(center_point[vb_index, 0], center_point[vb_index, 1], str(vb_index + 1),
                 size=10, alpha=0.9, color=color_list[vb_index % 5])

        center_p = self.make_point(index)

        # state_patch = self.make_state(index, center_p[vb_index])
        state_patch = self.make_state_v2(index, center_p[vb_index])

        f2 = plt.figure()
        plt.title('State patch (0)')
        plt.imshow(state_patch[0, :, :, 0])

        f3 = plt.figure()
        plt.title('State patch (1)')
        plt.imshow(state_patch[0, :, :, 1])
        plt.show()