# -*- coding: utf- 8 -*-

"""Attempt to train Unet."""

import os
import argparse
from utils import check_directories, initialize_script, end_script, jaccard2_loss, jaccard_metric, jaccard_metric_batch
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage import io
from models import model_on_steroids_densenet121

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def dataset_producer(path_input, path_mask):
    for el in os.listdir("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_input[0]):
        image_1, image_2, image_3 = io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_input[0] + el),\
                           io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_input[1] + el), \
                            io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_input[2] + el.split("ch")[0] + "ch1.tiff")
        image_1, image_2, image_3 = tf.convert_to_tensor(image_1), tf.convert_to_tensor(image_2), tf.convert_to_tensor(image_3)
        image = tf.stack([image_1, image_2, image_3], axis=2)[:, :, :, 0] #de taille (512, 512, 3)
        mask = io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_mask + el.split("ch")[0] + "ch4.png")
        mask = np.sign(mask) #convertir en semantic
        mask = tf.convert_to_tensor(mask)
        mask = tf.reshape(mask, [512, 512, 1])
        yield image, mask

if __name__ == "__main__":
    print()
    print("Run script unet_batch.py")
    print("TensorFlow version: {0}".format(tf.__version__))

    # get GPU devices
    gpu_devices = tf.config.experimental.list_logical_devices('GPU')
    print("Number of GPUs: ", len(gpu_devices))
    for device in gpu_devices:
        print("\r", device.name)
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory",
                        help="Path of the input directory.",
                        type=str)
    parser.add_argument("output_directory",
                        help="Path of the output directory.",
                        type=str)
    parser.add_argument("--log_directory",
                        help="Path of the log directory.",
                        type=str)

    # initialize parameters
    args = parser.parse_args()
    input_directory = args.input_directory
    output_directory = args.output_directory
    log_directory = args.log_directory

    # check directories exists
    check_directories([input_directory, output_directory, log_directory])

    # initialize script
    start_time, log_name = initialize_script(log_directory)

    # initialize Dataset from filename patterns
    dataset_train = tf.data.Dataset.from_generator(lambda : dataset_producer(["train/br_z3_train/", "train/br_z4_train/", "train/pc_train/"],
                                                                             "train/mask_train/"),
                                                   output_types=(tf.float32, tf.float32),
                                             output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_test = tf.data.Dataset.from_generator(lambda : dataset_producer(["test/br_z3_test/", "test/br_z4_test/", "test/pc_test/"],
                                                                            "test/mask_test/"),
                                                  output_types=(tf.float32, tf.float32),
                                             output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_val = tf.data.Dataset.from_generator(lambda : dataset_producer(["val/br_z3_val/", "val/br_z4_val/", "val/pc_val/"],
                                                                           "val/mask_val/"),
                                                 output_types=(tf.float32, tf.float32),
                                             output_shapes=((512, 512, 3), (512, 512, 1)))


    # Count Dataset elements
    train_size = 0
    test_size = 0
    val_size = 0

    for element in dataset_train:
        train_size += 1
    for element in dataset_test:
        test_size += 1
    for element in dataset_val:
        val_size += 1
    print("\r train: {0}".format(train_size))
    print("\r validation: {0}".format(val_size))
    print("\r test: {0}".format(test_size), "\n")

    # Define batches
    train_dataset = dataset_train.batch(8)
    validation_dataset = dataset_val.batch(1)
    test_dataset = dataset_test.batch(1)

    # Initialize and compile model

    # Loading weights to second model
    inputs = Input(shape=(512, 512, 3), name="input", dtype="float32")
    outputs = model_on_steroids_densenet121(inputs)
    model_0 = Model(inputs, outputs)
    model_0.load_weights(os.path.join("/mnt/data3/mphilbert/output/log/20201026104808_unet_on_steroids", "model_weights/cp.ckpt")) # loading the weights of pre-trained in silico labeling model
    model_0.summary()
    o = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same", name = "conv_sigm")(model_0.layers[-2].output)
    model = Model([model_0.input], [o])
    opt = Adam(lr=0.001)
    loss_func = jaccard2_loss
    model.compile(loss=loss_func, optimizer=opt)
    model.summary()


    #model.load_weights(
    #    os.path.join("/mnt/data3/mphilbert/output/log/20201027095443_isl_segmentation", "model_weights/cp.ckpt"))

    # Save model
    path_log_directory = os.path.join(log_directory, log_name)
    path_checkpoint_file = os.path.join(path_log_directory, "model_weights/cp.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint_file, save_weights_only=True, verbose=1)

    # Create a callback for visualization on tensorboard
    log_dir = os.path.join(path_log_directory, "visual/fit/")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback for Earlystopping
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # Fit model
    nb_epoch = 150
    fitting = model.fit(train_dataset, epochs=nb_epoch, validation_data=validation_dataset, callbacks=[cp_callback, tensorboard_callback, early_stop_callback])
    model.save(os.path.join(path_log_directory, 'my_model'))
    print("Best validation loss on initial data: {0:.5f}".format(np.min(fitting.history['val_loss'])), "\n")

    # Evaluate model : jaccard metric
    print("jaccard:")
    print("\r train: {0:.5f}".format(jaccard_metric_batch(train_dataset, model)))
    print("\r validation: {0:.5f}".format(jaccard_metric(validation_dataset, model)))
    print("\r test: {0:.5f}".format(jaccard_metric(test_dataset, model)))

    # Run the trained model on examples from the test dataset and plot the generated masks in a file named output_masks

    pred = model.predict(test_dataset, batch_size=4)
    pred[pred > 0.5] = 1 # from sigmoid to label
    pred[pred <= 0.5] = 0
    print(pred.shape)

    os.mkdir(os.path.join(path_log_directory, "output_masks/"))

    i=0
    for element in test_dataset :
        im, lab = element
        fig = plt.figure(figsize=(30, 30))
        ax1 = fig.add_subplot(5, 5, 1)
        plt.title("Input : brightfield z3")
        ax1.imshow(im[0, :, :, 0], cmap="gray")
        ax2 = fig.add_subplot(5, 5, 2)
        plt.title("Input : brightfield z4")
        ax2.imshow(im[0, :, :, 1], cmap="gray")
        ax3 = fig.add_subplot(5, 5, 3)
        plt.title("Input : pc")
        ax3.imshow(im[0, :, :, 2], cmap="gray")
        ax4 = fig.add_subplot(5, 5, 4)
        plt.title("Label : mask dapi")
        ax4.imshow(tf.reshape(lab, [512, 512]), cmap="gray")
        ax5 = fig.add_subplot(5, 5, 5)
        plt.title("Prediction : mask dapi")
        ax5.imshow(pred[i].reshape(512, 512), cmap="gray")
        plt.tight_layout()
        plt.savefig(os.path.join(path_log_directory, "output_masks/") + "fig" + str(i) + ".png")
        i += 1

    end_script(start_time)