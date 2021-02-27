# -*- coding: utf- 8 -*-

"""Attempt to train Unet."""

import os
import argparse
from utils import check_directories, initialize_script, end_script, dataset_producer, pearson_coeff, pearson_coeff_batch
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from models import unet_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    print()
    print("Run script unet_batch.py")
    print("TensorFlow version: {0}".format(tf.__version__))

    # Get GPU devices
    gpu_devices = tf.config.experimental.list_logical_devices('GPU')
    print("Number of GPUs: ", len(gpu_devices))
    for device in gpu_devices:
        print("\r", device.name)
    print()

    # Parse arguments
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

    # Initialize parameters
    args = parser.parse_args()
    input_directory = args.input_directory
    output_directory = args.output_directory
    log_directory = args.log_directory

    # Check directories exists
    check_directories([input_directory, output_directory, log_directory])

    # Initialize script
    start_time, log_name = initialize_script(log_directory)

    # Initialize Dataset from filename patterns
    dataset_train = tf.data.Dataset.from_generator(lambda : dataset_producer(["train/br_z3_train/", "train/br_z4_train/", "train/pc_train/"],
                                                                             "train/dapi_train/"),
                                                   output_types=(tf.float32, tf.float32), output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_test = tf.data.Dataset.from_generator(lambda : dataset_producer(["test/br_z3_test/", "test/br_z4_test/", "test/pc_test/"],
                                                                            "test/dapi_test/"),
                                                  output_types=(tf.float32, tf.float32), output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_val = tf.data.Dataset.from_generator(lambda : dataset_producer(["val/br_z3_val/", "val/br_z4_val/", "val/pc_val/"],
                                                                           "val/dapi_val/"),
                                                 output_types=(tf.float32, tf.float32), output_shapes=((512, 512, 3), (512, 512, 1)))


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
    inputs = Input(shape=(512, 512, 3), name="input", dtype="float32")
    outputs = unet_model(inputs)
    model = Model(inputs, outputs)
    opt = Adam(lr=0.001)
    loss_func = MeanAbsoluteError()
    model.compile(loss=loss_func, optimizer=opt)
    print(model.summary(), "\n")

    #model.load_weights(os.path.join("/mnt/data3/mphilbert/output/log/20201118171946_3_inputs_to_dapi", "model_weights/cp.ckpt"))

    # Save model
    path_log_directory = os.path.join(log_directory, log_name)
    path_checkpoint_file = os.path.join(path_log_directory, "model_weights/cp.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint_file, save_weights_only=True, verbose=1)

    # Create a callback for visualization on tensorboard
    log_dir = os.path.join(path_log_directory, "visual/fit/")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Create a callback for Earlystopping
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    # Fit model
    nb_epoch = 150
    fitting = model.fit(train_dataset, epochs=nb_epoch, validation_data=validation_dataset, callbacks=[cp_callback, tensorboard_callback, early_stop_callback])
    model.save(os.path.join(path_log_directory, 'my_model'))
    print("Best validation loss on initial data: {0:.5f}".format(np.min(fitting.history['val_loss'])), "\n")

    # Evaluate model : print the L1 loss
    print("L1 loss:")
    print("\r train: {0:.5f}".format(model.evaluate(train_dataset, batch_size = 4)))
    print("\r validation: {0:.5f}".format(model.evaluate(validation_dataset, batch_size = 4)))
    print("\r test: {0:.5f}".format(model.evaluate(test_dataset, batch_size = 4)))

    # Evaluate the model : print the mean Pearson correlation coefficient
    print("Pearson coeff:")
    print("\r train: {0:.5f}".format(pearson_coeff_batch(train_dataset, model)))
    print("\r validation: {0:.5f}".format(pearson_coeff(validation_dataset, model)))
    print("\r test: {0:.5f}".format(pearson_coeff(test_dataset, model)))

    # Run the trained model on examples from the test dataset and plot the generated images in a file named output_images
    pred = model.predict(test_dataset, batch_size = 4)
    print(pred.shape)

    os.mkdir(os.path.join(path_log_directory, "output_images/"))

    i=0
    for element in test_dataset :
        im, lab = element
        fig = plt.figure(figsize=(30, 30))
        ax1 = fig.add_subplot(4, 4, 1)
        plt.title("Input : Brightfield z3")
        ax1.imshow(im[0, :, :, 0], cmap="gray")
        ax2 = fig.add_subplot(4, 4, 2)
        plt.title("Input : Brightfield z4")
        ax2.imshow(im[0, :, :, 1], cmap="gray")
        ax3 = fig.add_subplot(4, 4, 3)
        plt.title("Label : dapi")
        ax3.imshow(tf.reshape(lab, [512, 512]), cmap ="gray")
        ax4 = fig.add_subplot(4, 4, 4)
        plt.title("Prediction : dapi")
        ax3.imshow(pred[i].reshape(512, 512), cmap="gray")
        plt.tight_layout()
        plt.savefig(os.path.join(path_log_directory, "output_images/") + "fig" + str(i) + ".png")
        i += 1


    end_script(start_time)
