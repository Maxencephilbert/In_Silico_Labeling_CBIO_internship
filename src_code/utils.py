# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""Routine functions."""

import os
import time
import datetime
import sys
import inspect
import shutil
import tensorflow as tf
from skimage import io
from scipy import stats
import numpy as np
from tensorflow.keras import backend as K

def dataset_producer(path_inputs, path_label):
    for el in os.listdir("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_inputs[0]):
            image_1, image_2, image_3 = io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_inputs[0] + el),\
                                             io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_inputs[1] + el),\
                                             io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_inputs[2] + el.split("ch")[0] + "ch1.tiff")
            image_1, image_2, image_3 = tf.convert_to_tensor(image_1), tf.convert_to_tensor(image_2), \
                                             tf.convert_to_tensor(image_3)
            image = tf.stack([image_1, image_2, image_3], axis=2)[:, :, :, 0] #de taille (512, 512, 3)
            label = io.imread("/mnt/data3/mphilbert/data/crop_dapi_segm/" + path_label + el.split("ch")[0] + "ch4.tiff")
            label = tf.convert_to_tensor(label)
            yield image, label


def pearson_coeff_batch(dataset, model):
    pred = model.predict(dataset, batch_size = 4)
    corr = 0
    i = 0
    for element in dataset:
        for j in range(element[0].numpy().shape[0]):
            corr += stats.pearsonr(pred[i].flatten(), element[1][j].numpy().flatten())[0]
            i+=1
    return corr/pred.shape[0]


def pearson_coeff(dataset, model):
    # calculates the pearson correlation coefficient : metric for our model
    corr = 0
    i = 0
    pred = model.predict(dataset, batch_size = 4)
    for element in dataset:
        im, lab = element
        #pred = model.predict(tf.reshape(im, [1, 512, 512, 1]))
        corr += stats.pearsonr(pred[i].flatten(), lab.numpy().flatten())[0]
        i+=1
    return corr/pred.shape[0]

SMOOTH = 1.0

def jaccard2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


def jaccard(im1, im2):
    union_vol = np.sum(np.maximum(im1, im2))
    if union_vol <= 0:
        return 1.0
        #raise ValueError("Images are empty (or contain negative values)")
    return(np.sum(np.minimum(im1, im2)) / union_vol)


def jaccard_metric(dataset, model):
    pred = model.predict(dataset, batch_size = 4)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    jacc = 0
    i = 0
    for element in dataset:
        im, lab = element
        jacc += jaccard(lab.numpy(), pred[i])
        i+=1
    return jacc/pred.shape[0]


def jaccard_metric_batch(dataset, model):
    pred = model.predict(dataset, batch_size = 4)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    jacc = 0
    i = 0
    for element in dataset:
        for j in range(element[0].numpy().shape[0]):
            jacc += jaccard(element[1][j].numpy(), pred[i])
            i+=1
    return jacc/pred.shape[0]


def check_directories(path_directories):
    # check directories exist
    #stack.check_parameter(path_directories=list)
    for path_directory in path_directories:
        if not os.path.isdir(path_directory):
            raise ValueError("Directory does not exist: {0}"
                             .format(path_directory))

    return


def initialize_script(log_directory, experiment_name=None):
    # check parameters
    #stack.check_parameter(log_directory=str)
    #stack.check_parameter(experiment_name=(str, type(None)))

    # get filename of the script that call this function
    try:
        previous_filename = inspect.getframeinfo(sys._getframe(1))[0]
    except ValueError:
        previous_filename = None

    # get date of execution
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2].split(" ")[0]
    hour = date.split(":")[0].split(" ")[1]
    minute = date.split(":")[1]
    second = date.split(":")[2]

    # format log name
    log_date = year + month + day + hour + minute + second
    if previous_filename is not None:
        operation = os.path.basename(previous_filename)
        operation = operation.split(".")[0]
        if experiment_name is not None:
            log_name = "{0}_{1}_{2}".format(
                log_date, operation, experiment_name)
        else:
            log_name = "{0}_{1}".format(log_date, operation)
    else:
        if experiment_name is not None:
            log_name = "{0}_{1}".format(log_date, experiment_name)
        else:
            log_name = "{0}".format(log_date)

    # initialize logging in a specific log directory
    path_log_directory = os.path.join(log_directory, log_name)
    os.mkdir(path_log_directory)
    path_log_file = os.path.join(path_log_directory, "log")
    sys.stdout = Logger(path_log_file)

    # copy python script in the log directory
    if previous_filename is not None:
        path_output = os.path.join(path_log_directory,
                                   os.path.basename(previous_filename))
        shutil.copyfile(previous_filename, path_output)

    # print information about launched script
    if previous_filename is not None:
        print("Running {0} file..."
              .format(os.path.basename(previous_filename)))
        print()
    start_time = time.time()
    if experiment_name is not None:
        print("Experiment name: {0}".format(experiment_name))
    print("Log directory: {0}".format(log_directory))
    print("Log name: {0}".format(log_name))
    print("Date: {0}".format(date), "\n")

    return start_time, log_name


def end_script(start_time):
    # check parameters
    #stack.check_parameter(start_time=(int, float))

    # time the script
    end_time = time.time()
    duration = int(round((end_time - start_time) / 60))
    print("Duration: {0} minutes".format(duration), "\n")

    return


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


