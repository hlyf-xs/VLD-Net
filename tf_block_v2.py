# from tensorflow.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Add
import tensorflow as tf


# Conv --- BN --- Relu
def conv_bn_ac(input, filter_num, kernel_size, strides, padding, dilation_rate, trainable, activation="relu"):

    conv = tf.compat.v1.layers.Conv2D(filters=filter_num,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      dilation_rate=dilation_rate,
                                      trainable=trainable)(input)

    bn = BatchNormalization()(conv)
    # bn = conv

    relu = Activation(activation=activation)(bn)

    return relu


# conv 3*3  conv 3*3
def standard_block(input, filter_num, kernel_size, strides, padding, dilation_rate, trainable, activation="relu"):

    conv1 = conv_bn_ac(input,
                       filter_num,
                       kernel_size,
                       strides,
                       padding,
                       dilation_rate,
                       trainable=trainable,
                       activation=activation)

    conv2 = conv_bn_ac(conv1,
                       filter_num,
                       kernel_size,
                       strides,
                       padding,
                       dilation_rate,
                       trainable=trainable,
                       activation=activation)

    return conv2


def residual_block(input, filter_num, kernel_size, strides, padding, dilation_rate, trainable, activation="relu"):

    conv0 = conv_bn_ac(input,
                       filter_num,
                       kernel_size,
                       strides,
                       padding,
                       dilation_rate,
                       trainable=trainable, activation=activation)

    conv1 = tf.compat.v1.layers.Conv2D(filters=filter_num,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   dilation_rate=dilation_rate,
                   trainable=trainable)(conv0)

    bn = BatchNormalization()(conv1)
    # bn = conv1

    relu = Activation(activation)(Add()([conv0, bn]))

    return relu


def conv_bn_relu(input, filter_num, kernel_size, strides, padding, use_bias, dilation_rate, trainable):

    conv = tf.compat.v1.layers.Conv2D(filters=filter_num,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=use_bias,
                                      dilation_rate=dilation_rate,
                                      trainable=trainable)(input)

    bn = BatchNormalization()(conv)

    relu = Activation(activation='relu')(bn)

    return relu


def conv_bn(input, filter_num, kernel_size, strides, padding, use_bias, dilation_rate, trainable):

    conv = tf.compat.v1.layers.Conv2D(filters=filter_num,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=use_bias,
                                      dilation_rate=dilation_rate,
                                      trainable=trainable)(input)

    bn = BatchNormalization()(conv)

    return bn
