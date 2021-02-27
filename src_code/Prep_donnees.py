import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import argparse
from skimage.measure import label
import os

def flip_left_right(img) :
    return tf.image.flip_left_right(img)


def flip_up_down(img):
    return tf.image.flip_up_down(img)


def rot90(img):
    return tf.image.rot90(img)


def identity(img):
    return img


def random_crop_image_and_labels(liste_images, h, w):
    # randomly crops the input image and its labels
    liste_output = []
    combined = tf.concat(liste_images, axis=2)
    combined_crop = tf.image.random_crop(combined, size=[h, w, len(liste_images)])
    for i in range(len(liste_images)):
        liste_output.append(combined_crop[:, :, i:i+1])
        liste_output[i].set_shape([h, w, 1])
    return liste_output


def augmentation():
    liste_operations = [identity, flip_left_right, flip_up_down, rot90]
    random_operation = np.random.choice(liste_operations)
    return random_operation


def norm(im):
    im = tf.cast(im, tf.float32)
    mean = tf.math.reduce_mean(im)
    std = tf.math.reduce_std(im)
    im_norm = (im - mean) / std
    return im_norm


def normalize(im, mean, std):
    im = tf.cast(im, tf.float32)
    im_norm = (im - mean) / std
    return im_norm

# parser

parser = argparse.ArgumentParser()

parser.add_argument("puit_name",
                    help="Name of the puit",
                    type=str)

# initialize parameters
args = parser.parse_args()
puit_name = args.puit_name


def rescale(image, min, max):
    image = tf.cast(image, tf.float32)
    image = image - min
    image = image/max
    return image


for element in open("/mnt/data3/mphilbert/data/dic_val.txt", "r"):
    if puit_name in element:
        DIC_0 = io.imread("/mnt/data3/mphilbert/data/dic_data/DIC_channel/z0/" + element.strip() + "chDIC.TIF").reshape(1024, 1024, 1)
        DIC_0 = tf.convert_to_tensor(DIC_0)
        DIC_0_norm = norm(DIC_0)

        DIC_1 = io.imread("/mnt/data3/mphilbert/data/dic_data/DIC_channel/z1/" + element.strip() + "chDIC.TIF").reshape(
            1024, 1024, 1)
        DIC_1 = tf.convert_to_tensor(DIC_1)
        DIC_1_norm = norm(DIC_1)

        DIC_2 = io.imread("/mnt/data3/mphilbert/data/dic_data/DIC_channel/z2/" + element.strip() + "chDIC.TIF").reshape(
            1024, 1024, 1)
        DIC_2 = tf.convert_to_tensor(DIC_2)
        DIC_2_norm = norm(DIC_2)

        DIC_3 = io.imread("/mnt/data3/mphilbert/data/dic_data/DIC_channel/z3/" + element.strip() + "chDIC.TIF").reshape(
            1024, 1024, 1)
        DIC_3 = tf.convert_to_tensor(DIC_3)
        DIC_3_norm = norm(DIC_3)

        DIC_4 = io.imread("/mnt/data3/mphilbert/data/dic_data/DIC_channel/z4/" + element.strip() + "chDIC.TIF").reshape(
            1024, 1024, 1)
        DIC_4 = tf.convert_to_tensor(DIC_4)
        DIC_4_norm = norm(DIC_4)

        H = io.imread("/mnt/data3/mphilbert/data/dic_data/Hoechst_channel/z4/" + element.strip() + "chH.TIF").reshape(1024, 1024, 1)
        H = tf.convert_to_tensor(H)
        H_norm = norm(H)

        Cy3 = io.imread("/mnt/data3/mphilbert/data/dic_data/Cy3_channel/z4/" + element.strip() + "chCy3.TIF").reshape(
            1024, 1024, 1)
        Cy3 = tf.convert_to_tensor(Cy3)
        Cy3_norm = norm(Cy3)

        GFP = io.imread("/mnt/data3/mphilbert/data/dic_data/GFP_channel/z4/" + element.strip() + "chGFP.TIF").reshape(
            1024, 1024, 1)
        GFP = tf.convert_to_tensor(GFP)
        GFP_norm = norm(GFP)

        Cy5 = io.imread("/mnt/data3/mphilbert/data/dic_data/Cy5_channel/z4/" + element.strip() + "chCy5.TIF").reshape(
            1024, 1024, 1)
        Cy5 = tf.convert_to_tensor(Cy5)
        Cy5_norm = norm(Cy5)

        #mask = io.imread(
        #    "/mnt/data3/mphilbert/data/mip/dapi/mask_maxence/good/nuc_mask/" + element.strip() + "ch4.png").reshape(
        #    2160, 2160, 1)
        #mask = tf.convert_to_tensor(mask, tf.float32)

        for i in range(10):
                liste_crop = random_crop_image_and_labels([DIC_0_norm, DIC_1_norm, DIC_2_norm, DIC_3_norm, DIC_4_norm, H_norm, Cy3_norm, GFP_norm, Cy5_norm], 512, 512)
                DIC_0_norm_cropped, DIC_1_norm_cropped, DIC_2_norm_cropped, DIC_3_norm_cropped, DIC_4_norm_cropped, H_norm_cropped, Cy3_norm_cropped, GFP_norm_cropped, \
                Cy5_norm_cropped = liste_crop[0], liste_crop[1], liste_crop[2], \
                                             liste_crop[3], liste_crop[4], liste_crop[5], liste_crop[6], liste_crop[7], liste_crop[8]
                random_op = augmentation()
                DIC_0_norm_augm, DIC_1_norm_augm, DIC_2_norm_augm, DIC_3_norm_augm, DIC_4_norm_augm, H_norm_augm, \
                Cy3_norm_augm, GFP_norm_augm, Cy5_norm_augm = random_op(DIC_0_norm_cropped),\
                                                              random_op(DIC_1_norm_cropped), random_op(DIC_2_norm_cropped),\
                                                              random_op(DIC_3_norm_cropped), random_op(DIC_4_norm_cropped),\
                                                              random_op(H_norm_cropped),random_op(Cy3_norm_cropped), \
                                                              random_op(GFP_norm_cropped), random_op(Cy5_norm_cropped)

                #mask_norm_augm label(tf.cast(mask_norm_augm, tf.bool).numpy())

                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/DIC_0_val/" + str(i) + "_" + element.strip() + "chDIC.TIF",
                          DIC_0_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/DIC_1_val/" + str(
                    i) + "_" + element.strip() + "chDIC.TIF",
                          DIC_1_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/DIC_2_val/" + str(
                    i) + "_" + element.strip() + "chDIC.TIF",
                          DIC_2_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/DIC_3_val/" + str(
                    i) + "_" + element.strip() + "chDIC.TIF",
                          DIC_3_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/DIC_4_val/" + str(
                    i) + "_" + element.strip() + "chDIC.TIF",
                          DIC_4_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/H_val/" + str(i) + "_" + element.strip() + "chH.TIF",
                          H_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/Cy3_val/" + str(
                    i) + "_" + element.strip() + "chCy3.TIF",
                          Cy3_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/GFP_val/" + str(
                    i) + "_" + element.strip() + "chGFP.TIF",
                          GFP_norm_augm.numpy())
                io.imsave("/mnt/data3/mphilbert/data/dic_crops/val/Cy5_val/" + str(
                    i) + "_" + element.strip() + "chCy5.TIF",
                          Cy5_norm_augm.numpy())

                fig = plt.figure(figsize=(30, 30))
                ax1 = fig.add_subplot(9, 9, 1)
                plt.title("DIC z0")
                ax1.imshow(tf.reshape(DIC_0_norm_augm, [512, 512]), cmap="gray")
                ax2 = fig.add_subplot(9, 9, 2)
                plt.title("DIC z1")
                ax2.imshow(tf.reshape(DIC_1_norm_augm, [512, 512]), cmap="gray")
                ax3 = fig.add_subplot(9, 9, 3)
                plt.title("DIC z2")
                ax3.imshow(tf.reshape(DIC_2_norm_augm, [512, 512]), cmap="gray")
                ax4 = fig.add_subplot(9, 9, 4)
                plt.title("DIC z3")
                ax4.imshow(tf.reshape(DIC_3_norm_augm, [512, 512]), cmap="gray")
                ax5 = fig.add_subplot(9, 9, 5)
                plt.title("DIC z4")
                ax5.imshow(tf.reshape(DIC_4_norm_augm, [512, 512]), cmap="gray")
                ax6 = fig.add_subplot(9, 9, 6)
                plt.title("Hoechst")
                ax6.imshow(tf.reshape(H_norm_augm, [512, 512]), cmap="gray")
                ax7 = fig.add_subplot(9, 9, 7)
                plt.title("Cy3")
                ax7.imshow(tf.reshape(Cy3_norm_augm, [512, 512]), cmap="gray")
                ax8 = fig.add_subplot(9, 9, 8)
                plt.title("GFP")
                ax8.imshow(tf.reshape(GFP_norm_augm, [512, 512]), cmap="gray")
                ax9 = fig.add_subplot(9, 9, 9)
                plt.title("Cy5")
                ax9.imshow(tf.reshape(Cy5_norm_augm, [512, 512]), cmap="gray")
                plt.tight_layout()

                plt.savefig(
                    "/mnt/data3/mphilbert/data/dic_crops/val/plot_val/" + element.strip() + "crop" + str(i) + ".png", bbox_inches='tight')








