import tensorflow as tf
import numpy as np
import os

from tf_block_v2 import standard_block, residual_block, conv_bn_relu, conv_bn
from tensorflow.keras.layers import Add, MaxPool2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation


# --------------------------  hyper parameters ----------------------------
MAX_EPISODES = 100000
MAX_EP_STEPS = 17

LR_A = 0.0001
LR_C = 0.0002

GAMMA = 0.95
TAU = 0.01

MEMORY_CAPACITY = 10000

BATCH_SIZE = 17
soft_replace_rate = 10

LOG_DIR = 'VB_logs/'
MODEL_SAVE_PATH = 'models/'


class Agent(object):

    def __init__(self, out_graph):

        self.out_graph = out_graph

        self.memory_a = np.zeros((MEMORY_CAPACITY, 2), dtype=np.float32)
        self.memory_s = np.zeros((MEMORY_CAPACITY, 200, 200, 2), dtype=np.float32)
        self.memory_s_ = np.zeros((MEMORY_CAPACITY, 200, 200, 2), dtype=np.float32)
        self.memory_r = np.zeros((MEMORY_CAPACITY, 1), dtype=np.float32)

        self.pointer = 0

        # test multi GPU train
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

        self.sess = tf.compat.v1.Session()

        tf.compat.v1.disable_eager_execution()

        self.S = tf.compat.v1.placeholder(tf.float32, [None, 200, 200, 2], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, 200, 200, 2], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        # build network
        with tf.compat.v1.variable_scope('Actor'):

            self.a = self._build_actor_net(self.S, scope='eval', trainable=True)

            a_ = self._build_actor_net(self.S_, scope='target', trainable=False)

        with tf.compat.v1.variable_scope('Critic'):

            q = self._build_critic_net(self.S, self.a, scope='eval', trainable=True)

            q_ = self._build_critic_net(self.S_, a_, scope='target', trainable=False)

        # get network parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.test_net_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                           scope='Test_Net/new_net')

        # target net replacement
        #  --- soft replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # training Critic network
        q_target = self.R + GAMMA * q_

        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)

        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        tf.compat.v1.summary.scalar('td_error', td_error)

        # training Actor network
        a_loss = - tf.reduce_mean(input_tensor=q)

        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        tf.compat.v1.summary.scalar('a_loss', a_loss)

        self.merged = tf.compat.v1.summary.merge_all()

        if self.out_graph:
            self.writer = tf.compat.v1.summary.FileWriter(LOG_DIR, self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.Actor_Net_Saver = tf.compat.v1.train.Saver(self.ae_params, max_to_keep=800)

    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :, :, :]})

        return action

    def store_transition(self, s, a, r, s_):

        index = self.pointer % MEMORY_CAPACITY

        self.memory_a[index, :] = a
        self.memory_s[index, :, :, :] = s
        self.memory_s_[index, :, :, :] = s_
        self.memory_r[index, :] = r

        self.pointer += 1

    def save_actor_net(self, step):

        self.Actor_Net_Saver.save(self.sess, save_path=MODEL_SAVE_PATH + str(step) + '.ckpt')
        print('\nSave Actor Network Weights !!!!')

    def load_actor_net(self, model_dir, step):

        self.Actor_Net_Saver.restore(self.sess, save_path=model_dir + str(step) + '.ckpt')
        print('\nLoad Actor Network Weights !!!')

    def learn(self, train_step):

        # if train_step > 0 and train_step % 7 == 0:
        if train_step > 0 and train_step % soft_replace_rate == 0:
            self.sess.run(self.soft_replace)
        # self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        batch_a = self.memory_a[indices, :]             # shape : (BATCH_SIZE, 2)             dtype : float32
        batch_s = self.memory_s[indices, :, :, :]       # shape : (BATCH_SIZE, 200, 400, 1)   dtype : float32
        batch_s_ = self.memory_s_[indices, :, :, :]     # shape : (BATCH_SIZE, 200, 400, 1)   dtype : float32
        batch_r = self.memory_r[indices, :]             # shape : (BATCH_SIZE, 1)             dtype : float32

        self.sess.run(self.atrain, {self.S: batch_s})
        self.sess.run(self.ctrain, {self.S: batch_s, self.a: batch_a, self.R: batch_r, self.S_: batch_s_})

        summary = self.sess.run(self.merged, feed_dict={self.S: batch_s, self.a: batch_a,
                                                        self.R: batch_r, self.S_: batch_s_})

        return summary

    def _build_actor_net(self, s, scope, trainable):

        with tf.compat.v1.variable_scope(scope):

            print('\n ############     Build Actor  {0}   Network !!!!!!!!!!!!!!!!  ######'.format(scope))

            # conv - bn - relu
            conv1 = tf.compat.v1.layers.Conv2D(filters=64,
                                               kernel_size=(3, 3),
                                               strides=(2, 2),
                                               padding="SAME",
                                               use_bias=False,
                                               trainable=trainable
                                               )(s)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation(activation='relu')(bn1)

            relu1 = Dropout(rate=0.2)(relu1)  # (200, 200, 64)
            relu1 = MaxPool2D(pool_size=2, strides=2)(relu1)
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
            block2_in1 = Dropout(rate=0.2)(block2_in1)
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
            block3_in1 = Dropout(rate=0.2)(block3_in1)
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
            block4_in1 = Dropout(rate=0.2)(block4_in1)
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

            fc1 = Flatten()(fc_block_in)
            print('\nfc1 ', fc1.shape)

            fc2 = tf.compat.v1.layers.Dense(4096, activation="relu", trainable=trainable)(fc1)
            print('\nfc2 ', fc2.shape)
            fc2 = Dropout(rate=0.2)(fc2)

            fc3 = tf.compat.v1.layers.Dense(512, activation="relu", trainable=trainable)(fc2)
            print('\nfc3 ', fc3.shape)
            fc3 = Dropout(rate=0.2)(fc3)

            fc4 = tf.compat.v1.layers.Dense(64, activation="relu", trainable=trainable)(fc3)
            print('\nfc4 ', fc4.shape)
            fc4 = Dropout(rate=0.2)(fc4)

            fc5 = tf.compat.v1.layers.Dense(16, activation="relu", trainable=trainable)(fc4)
            print('\nfc5 ', fc5.shape)
            fc5 = Dropout(rate=0.2)(fc5)

            fc6 = tf.compat.v1.layers.Dense(2, activation="tanh", trainable=trainable)(fc5)
            print('\nfc6 ', fc6.shape)
            print('\nfc6 ', fc6)

            return fc6

    def _build_critic_net(self, s, a, scope, trainable):

        with tf.compat.v1.variable_scope(scope):
            print('\n ############     Build Critic  {0}   Network   ######'.format(scope))

            # conv - bn - relu
            conv1 = tf.compat.v1.layers.Conv2D(filters=64,
                                               kernel_size=(3, 3),
                                               strides=(2, 2),
                                               padding="SAME",
                                               use_bias=False,
                                               trainable=trainable
                                               )(s)
            bn1 = BatchNormalization()(conv1)
            relu1 = Activation(activation='relu')(bn1)

            relu1 = Dropout(rate=0.2)(relu1)  # (200, 200, 64)
            relu1 = MaxPool2D(pool_size=2, strides=2)(relu1)
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
            block2_in1 = Dropout(rate=0.2)(block2_in1)
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
            block3_in1 = Dropout(rate=0.2)(block3_in1)
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
            block4_in1 = Dropout(rate=0.2)(block4_in1)
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

            fc1 = Flatten()(fc_block_in)
            print('\nfc1 ', fc1.shape)

            fc2 = tf.compat.v1.layers.Dense(4096, activation="relu", trainable=trainable)(fc1)
            print('\nfc2 ', fc2.shape)
            fc2 = Dropout(rate=0.2)(fc2)

            fc2_a = tf.concat([fc2, a], 1)
            print('\nfc2 + a ', fc2_a.shape)

            fc3 = tf.compat.v1.layers.Dense(512, activation="relu", trainable=trainable)(fc2_a)
            print('\nfc3 ', fc3.shape)
            fc3 = Dropout(rate=0.2)(fc3)

            fc4 = tf.compat.v1.layers.Dense(64, activation="relu", trainable=trainable)(fc3)
            print('\nfc4 ', fc4.shape)
            fc4 = Dropout(rate=0.2)(fc4)

            fc5 = tf.compat.v1.layers.Dense(16, activation="relu", trainable=trainable)(fc4)
            print('\nfc5 ', fc5.shape)
            fc5 = Dropout(rate=0.2)(fc5)

            fc6 = tf.compat.v1.layers.Dense(1, trainable=trainable)(fc5)
            print('\nfc6 ', fc6.shape)

            return fc6

