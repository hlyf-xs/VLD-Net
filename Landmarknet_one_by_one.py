from tensorflow.keras.layers import Add, MaxPool2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tf_block_v2 import conv_bn_relu, conv_bn
import tensorflow as tf

import tensorflow as tf
import numpy as np
import math
import os
import cv2


LOG_DIR = '../Landmark_Reg_1by1/Logs/'
MODEL_SAVE_DIR = '../Landmark_Reg_1by1/model_weights/'


learning_rate = 0.0005
batch_size = 20

IMG_SAVE_DIR = '../dataset/vb_patch_dataset/TrainSet/aug_patch_img/'
img_dirs = os.listdir(IMG_SAVE_DIR)


def data_aug(ori_img, ori_landmark, angle, shift, scale):

    # [begin]---------  rotation  -----------
    img = rotate_img(ori_img, angle)
    landmark = get_rotate_landmark(ori_landmark, ori_img, -angle)
    # [end] ---------  rotation  -----------

    # [begin] ----------  shit -----------
    if shift == 0:      # shift up
        img = np.concatenate((img[scale:, :, :], img[:scale, :, :]), axis=0)
    elif shift == 1:    # shift down
        img = np.concatenate((img[750-scale:, :, :], img[:750-scale, :, :]), axis=0)
    elif shift == 2:    # shift left
        img = np.concatenate((img[:, scale:, :], img[:, :scale, :]), axis=1)
    elif shift == 3:    # shift right
        img = np.concatenate((img[:, 250-scale:, :], img[:, :250-scale, :]), axis=1)

    landmark = get_shift_landmark(landmark, shift, scale)
    # [end]  ----------  shit -----------

    return img, landmark


def rotate_img(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    rotated =cv2.warpAffine(image, M, (w, h))

    return rotated


def get_point(x, y, image, angle):

    h = image.shape[0]
    w = image.shape[1]

    (cX, cY) = (w // 2, h // 2)

    x = x
    y = h - y
    cX = cX
    cY = h - cY
    new_x = (x - cX) * math.cos(math.pi / 180.0 * angle) - (y - cY) * math.sin(math.pi / 180.0 * angle) + cX
    new_y = (x - cX) * math.sin(math.pi / 180.0 * angle) + (y - cY) * math.cos(math.pi / 180.0 * angle) + cY
    new_x = new_x
    new_y = h - new_y
    return round(new_x), round(new_y)


def get_rotate_landmark(landmark, image, angle):

    landmark_new = np.zeros((4, 2), dtype='float32')
    for i in range(4):

        x, y = landmark[i, 0], landmark[i, 1]
        new_x, new_y = get_point(x, y, image, angle)
        landmark_new[i, 0] = new_x
        landmark_new[i, 1] = new_y

    return landmark_new


def get_shift_landmark(landmark, shift, scale):

    landmark_new = np.zeros((4, 2), dtype='float32')
    for i in range(4):
        x, y = landmark[i, 0], landmark[i, 1]
        if shift == 0:
            new_x = x
            new_y = y - scale
        elif shift == 1:
            new_x = x
            new_y = y + scale
        elif shift == 2:
            new_x = x - scale
            new_y = y
        elif shift == 3:
            new_x = x + scale
            new_y = y

        landmark_new[i, 0] = new_x
        landmark_new[i, 1] = new_y

    return landmark_new


IMG_SAVE_DIR = '../dataset/vb_patch_dataset/TrainSet/aug_patch_img/'
img_dirs = os.listdir(IMG_SAVE_DIR)


def load_patch_img(index):
    img_path_name = os.path.join(IMG_SAVE_DIR, img_dirs[index])

    ori_img = cv2.imread(img_path_name)

    return ori_img


GT_SAVE_DIR = '../dataset/vb_patch_dataset/TrainSet/aug_GT_landmark/'


def load_patch_landmark(index):

    gt_landmark_name = img_dirs[index].split(".")[0]
    gt_path_name = os.path.join(GT_SAVE_DIR, gt_landmark_name) + '.npy'
    gt_landmark = np.load(gt_path_name)  # (4, 2)

    return gt_landmark


def load_img_and_landmark(index):

    IMG_SAVE_DIR = '../dataset/vb_patch_dataset/TrainSet/aug_patch_img/'
    GT_SAVE_DIR = '../dataset/vb_patch_dataset/TrainSet/aug_GT_landmark/'

    img_dirs = os.listdir(IMG_SAVE_DIR)
    img_path_name = os.path.join(IMG_SAVE_DIR, img_dirs[index])
    ori_img = cv2.imread(img_path_name)


    gt_landmark_name = img_dirs[index].split(".")[0]
    gt_path_name = os.path.join(GT_SAVE_DIR, gt_landmark_name) + '.npy'
    gt_landmark = np.load(gt_path_name)  # (4, 2)


    aug = False
    if aug:

        angle = np.random.uniform(low=-6.0, high=6.0)
        shift = np.random.randint(0, 4)  # 0: up ;    1: down  ;   2: left     ;   3: right
        scale = np.random.randint(0, 5)

        ori_img, gt_landmark = data_aug(ori_img, gt_landmark, angle, shift, scale)

        add_gaussian_noise = False
        if add_gaussian_noise:
            ori_img = ori_img / 255.0
            mean = 0
            var = 0.00008
            noise = np.random.normal(mean, var ** 0.5, ori_img.shape)

            gaussian_out = ori_img + noise
            gaussian_out = np.clip(gaussian_out, 0, 1)
            ori_img = np.uint8(gaussian_out * 255)

    return ori_img, gt_landmark


class LandmarkNet(object):

    def __init__(self, out_graph=False,):

        self.out_graph = out_graph
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # choose gpu
        self.batch_size = batch_size

        self.sess = tf.compat.v1.Session()
        tf.compat.v1.disable_eager_execution()

        # Input_tensor :  VB patch image     GT_Landmark : Scaled Landmarks
        self.Input_tensor = tf.compat.v1.placeholder(tf.float32, [None, 100, 100, 1], 'input_tensor')
        self.GT_tensor = tf.compat.v1.placeholder(tf.float32, [None, 8], 'gt_landmarks')

        with tf.compat.v1.variable_scope('reg_net'):
            self.landmarks = self._build_landmarknet(self.Input_tensor, scope='landmarknet', trainable=True)

        # get network parameters
        self.landmarknet_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                              scope='reg_net/landmarknet')

        # training landmark net
        self.mse_loss = tf.compat.v1.losses.mean_squared_error(labels=self.GT_tensor,
                                                               predictions=self.landmarks)

        self.loss2 = tf.compat.v1.reduce_mean(tf.square(self.GT_tensor - self.landmarks))


        tf.compat.v1.summary.scalar('mse', self.mse_loss)

        self.global_ = tf.Variable(tf.constant(0))
        self.lr = tf.compat.v1.train.exponential_decay(learning_rate=learning_rate,
                                                       global_step=self.global_,
                                                       decay_steps=150000,
                                                       decay_rate=0.8,
                                                       staircase=True)

        self.train = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.mse_loss,
                                                                        global_step=self.global_,
                                                                        var_list=self.landmarknet_params)
        self.merged = tf.compat.v1.summary.merge_all()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Tensorboard Log
        if True:
            self.writer = tf.compat.v1.summary.FileWriter(LOG_DIR, self.sess.graph)

        self.Actor_Saver = tf.compat.v1.train.Saver(self.landmarknet_params, max_to_keep=500)

    # ******************      input_tensor ---> [landmark net] ---> pre_landmarks
    def pre_landmarknet(self, input_tensor):

        pre_landmark = self.sess.run(self.landmarks, {self.Input_tensor: input_tensor[np.newaxis, :, :, :]})

        return pre_landmark

    def _build_landmarknet(self, input_img, scope, trainable):

        with tf.compat.v1.variable_scope(scope):
            print('\n ##########  Build Landmark Net #########'.format(scope))
            # conv - bn - relu
            conv1 = tf.compat.v1.layers.Conv2D(filters=64,
                                               kernel_size=(3, 3),
                                               strides=(2, 2),
                                               padding="SAME",
                                               use_bias=False,
                                               trainable=trainable
                                               )(input_img)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation(activation='relu')(bn1)

            relu1 = Dropout(rate=0.5)(relu1)
            print('\nrelu1 ', relu1.shape)

            # ****************************           block1 :  4 conv
            block1_out1 = conv_bn_relu(input=relu1, filter_num=64, kernel_size=(3, 3), strides=(1, 1),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock1_out1 ', block1_out1.shape)

            block1_out2 = conv_bn(input=block1_out1, filter_num=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock1_out2 ', block1_out2.shape)

            block1_in3 = Activation(activation='relu')(Add()([relu1, block1_out2]))

            block1_out3 = conv_bn_relu(block1_in3, filter_num=64, kernel_size=(3, 3), strides=(1, 1),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock1_out3 ', block1_out3.shape)

            block1_out4 = conv_bn(block1_out3, filter_num=64, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock1_out4 ', block1_out4.shape)

            block2_in1 = Activation(activation='relu')(Add()([block1_in3, block1_out4]))

            # *****************              block2
            block2_in1 = Dropout(rate=0.5)(block2_in1)
            block2_out1 = conv_bn_relu(block2_in1, filter_num=128, kernel_size=(3, 3), strides=(2, 2),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock2_out1 ', block2_out1.shape)

            block2_out2 = conv_bn(block2_out1, filter_num=128, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock2_out2 ', block2_out2.shape)

            block2_shortcut = conv_bn(block2_in1, filter_num=128, kernel_size=(1, 1), strides=(2, 2),
                                      padding="SAME", use_bias=False, dilation_rate=1,
                                      trainable=trainable)

            block2_in3 = Activation(activation='relu')(Add()([block2_out2, block2_shortcut]))

            block2_out3 = conv_bn_relu(block2_in3, filter_num=128, kernel_size=(3, 3), strides=(1, 1),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock2_out3 ', block2_out3.shape)

            block2_out4 = conv_bn(block2_out3, filter_num=128, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock2_out4 ', block2_out4.shape)

            block3_in1 = Activation(activation='relu')(Add()([block2_in3, block2_out4]))

            # *****************              block3
            block3_in1 = Dropout(rate=0.5)(block3_in1)
            block3_out1 = conv_bn_relu(block3_in1, filter_num=256, kernel_size=(3, 3), strides=(2, 2),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock3_out1 ', block3_out1.shape)

            block3_out2 = conv_bn(block3_out1, filter_num=256, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock3_out2 ', block3_out2.shape)

            block3_shortcut = conv_bn(block3_in1, filter_num=256, kernel_size=(1, 1), strides=(2, 2),
                                      padding="SAME", use_bias=False, dilation_rate=1,
                                      trainable=trainable)

            block3_in3 = Activation(activation='relu')(Add()([block3_out2, block3_shortcut]))

            block3_out3 = conv_bn_relu(block3_in3, filter_num=256, kernel_size=(3, 3), strides=(1, 1),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock3_out3 ', block3_out3.shape)

            block3_out4 = conv_bn(block3_out3, filter_num=256, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock3_out4 ', block3_out4.shape)

            block4_in1 = Activation(activation='relu')(Add()([block3_in3, block3_out4]))

            # *****************              block4
            block4_in1 = Dropout(rate=0.5)(block4_in1)
            block4_out1 = conv_bn_relu(block4_in1, filter_num=512, kernel_size=(3, 3), strides=(2, 2),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock4_out1 ', block4_out1.shape)

            block4_out2 = conv_bn(block4_out1, filter_num=512, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock4_out2 ', block4_out2.shape)

            block4_shortcut = conv_bn(block4_in1, filter_num=512, kernel_size=(1, 1), strides=(2, 2),
                                      padding="SAME", use_bias=False, dilation_rate=1,
                                      trainable=trainable)

            block4_in3 = Activation(activation='relu')(Add()([block4_out2, block4_shortcut]))

            block4_out3 = conv_bn_relu(block4_in3, filter_num=512, kernel_size=(3, 3), strides=(1, 1),
                                       padding="SAME", use_bias=False, dilation_rate=1,
                                       trainable=trainable)
            print('\nblock4_out3 ', block4_out3.shape)

            block4_out4 = conv_bn(block4_out3, filter_num=512, kernel_size=(3, 3), strides=(1, 1),
                                  padding="SAME", use_bias=False, dilation_rate=1,
                                  trainable=trainable)
            print('\nblock4_out4 ', block4_out4.shape)

            fc_block_in = Activation(activation='relu')(Add()([block4_in3, block4_out4]))

            fc = Flatten()(fc_block_in)
            print('\nfc ', fc.shape)

            fc1 = tf.compat.v1.layers.Dense(4096, activation='relu', trainable=trainable)(fc)
            fc1 = Dropout(rate=0.5)(fc1)
            print('\nfc1 ', fc1.shape)

            fc2 = tf.compat.v1.layers.Dense(1024, activation='relu', trainable=trainable)(fc1)
            fc2 = Dropout(rate=0.5)(fc2)
            print('\nfc2 ', fc2.shape)

            fc3 = tf.compat.v1.layers.Dense(512, activation='relu', trainable=trainable)(fc2)
            fc3 = Dropout(rate=0.5)(fc3)
            print('\nfc3 ', fc3.shape)

            fc4 = tf.compat.v1.layers.Dense(256, activation='relu', trainable=trainable)(fc3)
            fc4 = Dropout(rate=0.5)(fc4)
            print('\nfc4 ', fc4.shape)

            # fc5 (output)      # (1, 8)
            fc5 = tf.compat.v1.layers.Dense(8, activation='sigmoid', trainable=trainable)(fc4)
            print('\nfc5 ', fc5.shape)

            return fc5

    def learn(self, train_step, batch_input_tensor, batch_gt_landmark):

        lr = self.sess.run(self.lr, feed_dict={self.global_: train_step})

        print('train_step : {0} - lr : {1:.8f}'.format(train_step, lr))

        self.sess.run(self.train, feed_dict={self.Input_tensor: batch_input_tensor,
                                             self.GT_tensor: batch_gt_landmark})

        summary = self.sess.run(self.merged, feed_dict={self.Input_tensor: batch_input_tensor,
                                                        self.GT_tensor: batch_gt_landmark})

        return summary

    def save_network_weights(self, step):

        self.Actor_Saver.save(self.sess, save_path=MODEL_SAVE_DIR + str(step) + ".ckpt")
        print("Save  Network Weights !!! ")

    def load_network_weights(self, model_dir, step):

        self.Actor_Saver.restore(self.sess, save_path=model_dir + str(step) + ".ckpt")
        print("Load  Network Weights !!! ")

    def img_to_input_tensor(self, img_idx=0):
        # Input_tensor = np.ndarray((100, 100, 1), dtype=np.uint8)
        img = load_patch_img(img_idx)
        # print("img:", img.shape)
        # img, _ = load_img_and_landmark(img_idx)

        Input_tensor = img[:, :, 0:1]
        Input_tensor = Input_tensor.astype('float32') / 255.0  # (100, 100, 1) float32

        return Input_tensor

    def get_batch_img_tensor(self, batch_idx):

        batch_input = np.zeros((self.batch_size, 100, 100, 1), dtype=np.float32)
        for j in range(self.batch_size):
            batch_input[j:j + 1, :] = self.img_to_input_tensor(batch_idx * batch_size + j)

        return batch_input

    def label_to_ouput_tensor(self, img_idx=0):

        landmark = load_patch_landmark(img_idx)  # (4, 2)

        landmark = landmark / 100.0

        GT_tensor = np.zeros((8,), dtype='float32')
        GT_tensor[0:4] = landmark[:, 0]
        GT_tensor[4:8] = landmark[:, 1]

        return GT_tensor

    def get_batch_gt_tensor(self, batch_idx):

        batch_landmark = np.zeros((self.batch_size, 8), dtype=np.float32)
        for j in range(self.batch_size):
            batch_landmark[j:j + 1, :] = self.label_to_ouput_tensor(batch_idx * batch_size + j)

        return batch_landmark



