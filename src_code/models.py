from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.applications import DenseNet121
import tensorflow as tf


def unet_model(input_tensor):
    # contraction 1§l
    conv_1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv_1_1")(input_tensor)
    activ_1_1 = Activation("relu")(BatchNormalization()(conv_1_1))
    conv_1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv_1_2")(activ_1_1)
    activ_1_2 = Activation("relu")(BatchNormalization()(conv_1_2))
    conv_1_3 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same", name="conv_1_3")(activ_1_2)
    activ_1_3 = Activation("relu")(BatchNormalization()(conv_1_3))
    activ_1_3 = Dropout(0.25)(activ_1_3)
    # (?, 256, 256, 32)

    # contraction 2
    conv_2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_2_1")(activ_1_3)
    activ_2_1 = Activation("relu")(BatchNormalization()(conv_2_1))
    conv_2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_2_2")(activ_2_1)
    activ_2_2 = Activation("relu")(BatchNormalization()(conv_2_2))
    conv_2_3 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="same", name="conv_2_3")(activ_2_2)
    activ_2_3 = Activation("relu")(BatchNormalization()(conv_2_3))
    activ_2_3 = Dropout(0.5)(activ_2_3)
    # (?, 128, 128, 64)

    # contraction 3
    conv_3_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_3_1")(activ_2_3)
    activ_3_1 = Activation("relu")(BatchNormalization()(conv_3_1))
    conv_3_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_3_2")(activ_3_1)
    activ_3_2 = Activation("relu")(BatchNormalization()(conv_3_2))
    conv_3_3 = Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="same", name="conv_3_3")(activ_3_2)
    activ_3_3 = Activation("relu")(BatchNormalization()(conv_3_3))
    activ_3_3 = Dropout(0.5)(activ_3_3)
    # (?, 64, 64, 128)

    # contraction 4
    conv_4_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="conv_4_1")(activ_3_3)
    activ_4_1 = Activation("relu")(BatchNormalization()(conv_4_1))
    conv_4_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="conv_4_2")(activ_4_1)
    activ_4_2 = Activation("relu")(BatchNormalization()(conv_4_2))
    conv_4_3 = Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="same", name="conv_4_3")(activ_4_2)
    activ_4_3 = Activation("relu")(BatchNormalization()(conv_4_3))
    activ_4_3 = Dropout(0.5)(activ_4_3)
    # (?, 32, 32, 256)

    # bottom
    conv_5_1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", name="conv_5_1")(activ_4_3)
    activ_5_1 = Activation("relu")(BatchNormalization()(conv_5_1))
    conv_5_2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", name="conv_5_2")(
        activ_5_1)
    activ_5_2 = Activation("relu")(BatchNormalization()(conv_5_2))
    # (?, 32, 32, 512)

    # expansion 1
    upconv_6_1 = Conv2DTranspose(filters=256, kernel_size = (2, 2), strides = (2, 2), padding = "same", name = "upconv_6_1")(activ_5_2)
    upactiv_6_1 = Activation("relu")(BatchNormalization()(upconv_6_1))
    concat_6 = tf.concat(values=[activ_4_2, upactiv_6_1], axis=-1, name='concat_6')
    concat_6 = Dropout(0.5)(concat_6)
    conv_6_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="conv_6_1")(concat_6)
    activ_6_1 = Activation("relu")(BatchNormalization()(conv_6_1))
    conv_6_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", name="conv_6_2")(
        activ_6_1)
    activ_6_2 = Activation("relu")(BatchNormalization()(conv_6_2))
    activ_6_2 = Dropout(0.25)(activ_6_2)
    # (?, 64, 64, 256)

    # expansion 2
    upconv_7_1 = Conv2DTranspose(filters=128, kernel_size = (2, 2), strides = (2, 2), padding = "same", name = "upconv_7_1")(activ_6_2)
    upactiv_7_1 = Activation("relu")(BatchNormalization()(upconv_7_1))
    concat_7 = tf.concat(values=[activ_3_2, upactiv_7_1], axis=-1, name='concat_7')
    conv_7_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_7_1")(concat_7)
    activ_7_1 = Activation("relu")(BatchNormalization()(conv_7_1))
    conv_7_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="conv_7_2")(
        activ_7_1)
    activ_7_2 = Activation("relu")(BatchNormalization()(conv_7_2))
    # (?, 128, 128, 128)

    # expansion 3
    upconv_8_1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="same",
                                 name="upconv_8_1")(activ_7_2)
    upactiv_8_1 = Activation("relu")(BatchNormalization()(upconv_8_1))
    concat_8 = tf.concat(values=[activ_2_2, upactiv_8_1], axis=-1, name='concat_8')
    concat_8 = Dropout(0.5)(concat_8)
    conv_8_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_8_1")(concat_8)
    activ_8_1 = Activation("relu")(BatchNormalization()(conv_8_1))
    conv_8_2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="conv_8_2")(
        activ_8_1)
    activ_8_2 = Activation("relu")(BatchNormalization()(conv_8_2))
    # (?, 256, 256, 64)

    # expansion 4
    upconv_9_1 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="same",
                                 name="upconv_9_1")(activ_8_2)
    upactiv_9_1 = Activation("relu")(BatchNormalization()(upconv_9_1))
    concat_9 = tf.concat(values=[activ_1_2, upactiv_9_1], axis=-1, name='concat_9')
    conv_9_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", name="conv_9_1")(concat_9)
    activ_9_1 = Activation("relu")(BatchNormalization()(conv_9_1))
    # (?, 512, 512, 32)

    # final
    conv_10 = Conv2D(filters=1, kernel_size=(1, 1), activation=None, padding="same", name="conv_10")(
        activ_9_1)
    # (?, 512, 512, 1)
    return conv_10


def model_on_steroids_densenet121(input_tensor):

    densenet = DenseNet121(include_top=False, input_tensor=input_tensor,
                                              input_shape=(512, 512, 3))  # summarize the model

    for layer in densenet.layers:
        layer.trainable = True #Or True

    # expansion 1
    upconv_6_1 = Conv2DTranspose(filters=320, kernel_size=(2, 2), strides=(2, 2), padding="same",
                                 kernel_initializer='he_normal', name="upconv_6_1")(
        densenet.layers[-1].output)  # (32, 32, 320)
    upactiv_6_1 = Activation("relu")(BatchNormalization()(upconv_6_1))
    concat_6 = tf.concat(values=[densenet.get_layer("conv4_block24_concat").output, upactiv_6_1], axis=-1, name='concat_6')
    concat_6 = Dropout(0.5)(concat_6)
    conv_6_1 = Conv2D(filters=320, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_6_1")(concat_6)
    activ_6_1 = Activation("relu")(BatchNormalization()(conv_6_1))
    conv_6_2 = Conv2D(filters=320, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_6_2")(
        activ_6_1)
    activ_6_2 = Activation("relu")(BatchNormalization()(conv_6_2))
    activ_6_2 = Dropout(0.25)(activ_6_2)
    # (?, 32, 32, 320)

    # expansion 2
    upconv_7_1 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="same",  kernel_initializer='he_normal', name="upconv_7_1")(
        activ_6_2)  # (64,64,256)
    upactiv_7_1 = Activation("relu")(BatchNormalization()(upconv_7_1))
    concat_7 = tf.concat(values=[densenet.get_layer("conv3_block12_concat").output, upactiv_7_1], axis=-1, name='concat_7')
    conv_7_1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_7_1")(concat_7)
    activ_7_1 = Activation("relu")(BatchNormalization()(conv_7_1))
    conv_7_2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer='he_normal', name="conv_7_2")(
        activ_7_1)
    activ_7_2 = Activation("relu")(BatchNormalization()(conv_7_2))
    # (?, 64, 64, 256)

    # expansion 3
    upconv_8_1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="same",  kernel_initializer='he_normal',
                                 name="upconv_8_1")(activ_7_2)  # (128, 128, 256)
    upactiv_8_1 = Activation("relu")(BatchNormalization()(upconv_8_1))
    concat_8 = tf.concat(values=[densenet.get_layer("conv2_block6_concat").output, upactiv_8_1], axis=-1, name='concat_8')
    concat_8 = Dropout(0.5)(concat_8)
    conv_8_1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_8_1")(concat_8)
    activ_8_1 = Activation("relu")(BatchNormalization()(conv_8_1))
    conv_8_2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_8_2")(
        activ_8_1)
    activ_8_2 = Activation("relu")(BatchNormalization()(conv_8_2))
    # (?, 128, 128, 128)

    # expansion 4
    upconv_9_1 = Conv2DTranspose(filters=96, kernel_size=(2, 2), strides=(2, 2), padding="same",  kernel_initializer='he_normal',
                                 name="upconv_9_1")(activ_8_2)  # (256, 256, 64)
    upactiv_9_1 = Activation("relu")(BatchNormalization()(upconv_9_1))
    concat_9 = tf.concat(values=[densenet.get_layer("conv1/relu").output, upactiv_9_1], axis=-1, name='concat_9')
    concat_9 = Dropout(0.5)(concat_9)
    conv_9_1 = Conv2D(filters=96, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_9_1")(concat_9)
    activ_9_1 = Activation("relu")(BatchNormalization()(conv_9_1))
    conv_9_2 = Conv2D(filters=96, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_9_2")(
        activ_9_1)
    activ_9_2 = Activation("relu")(BatchNormalization()(conv_9_2))
    # (?, 256, 256, 96)

    # expansion 5
    upconv_10_1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="same", kernel_initializer='he_normal', # ou filters = 32 ?
                                  name="upconv_10_1")(activ_9_2)  # (512, 512, 64)
    upactiv_10_1 = Activation("relu")(BatchNormalization()(upconv_10_1))
    conv_10_1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same",  kernel_initializer='he_normal', name="conv_10_1")(upactiv_10_1)
    activ_10_1 = Activation("relu")(BatchNormalization()(conv_10_1))
    # (?, 512, 512, 64)

    # final
    conv_10 = Conv2D(filters=1, kernel_size=(1, 1), activation=None, name="conv_10")(
        activ_10_1)
    # (?, 512, 512, 1)
    return conv_10
