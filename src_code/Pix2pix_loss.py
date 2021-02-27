import time
import os
import argparse
from utils import check_directories, initialize_script, end_script, dataset_producer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Dropout
from scipy import stats
import matplotlib.pyplot as plt
import datetime


def pearson_coeff(dataset):
    # calculates the pearson correlation coefficient : metric for our model
    corr = 0
    i = 0
    for inp, tar in dataset:
        pred = generator(inp, training=True)
        corr += stats.pearsonr(pred[0].numpy().flatten(), tar.numpy().flatten())[0]
        i += 1
    return corr / i


def pearson_coeff_batch(dataset):
    corr = 0
    i = 0
    for element in dataset:
        for j in range(element[0].numpy().shape[0]):
            pred = generator(tf.reshape(element[0][j], [1, 512, 512, 3]), training=True)
            corr += stats.pearsonr(pred[0].numpy().flatten(), element[1][j].numpy().flatten())[0]
            i += 1
    return corr / i



def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator_unet():
    input_tensor = Input(shape=(512, 512, 3), name="input", dtype="float32")

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
    upconv_6_1 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="same", name="upconv_6_1")(
        activ_5_2)
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
    upconv_7_1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="same", name="upconv_7_1")(
        activ_6_2)
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
    conv_10 = Conv2D(filters=1, kernel_size=(3, 3), activation=None, padding="same", name="conv_10")(
        activ_9_1)
    # (?, 512, 512, 1)

    return Model(inputs=input_tensor, outputs=conv_10)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image', dtype="float32")
    tar = tf.keras.layers.Input(shape=[512, 512, 1], name='target_image', dtype="float32")

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 512, 512, 2)

    down1 = downsample(64, 4, False)(x)  # (bs, 256, 256, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 128, 128, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 64, 64, 256)
    down4 = downsample(512, 4)(down3)  # (bs, 32, 32, 512)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (bs, 34, 34, 512)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0, :, :, 0], test_input[0, :, :, 1], test_input[0, :, :, 2], tar[0, :, :, 0], prediction[0, :, :, 0]]
    title = ['Input Image : Brightfield z3', 'Input Image : Brightfield z4',  'Input Image : Phase contrast', 'Ground Truth Dapi', 'Predicted Image Dapi']

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(tf.reshape(display_list[i], [512, 512]), cmap = "gray")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print()
    print("Run script pix2pix_loss.py")
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
    dataset_train = tf.data.Dataset.from_generator(lambda : dataset_producer(["train/br_z3_train/", "train/br_z4_train/",
                                                                              "train/pc_train/"],
                                                                             "train/dapi_train/"),
                                                   output_types=(tf.float32, tf.float32), output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_test = tf.data.Dataset.from_generator(lambda : dataset_producer(["test/br_z3_test/", "test/br_z4_test/",
                                                                             "test/pc_test/"],
                                                                            "test/dapi_test/"),
                                                  output_types=(tf.float32, tf.float32), output_shapes=((512, 512, 3), (512, 512, 1)))

    dataset_val = tf.data.Dataset.from_generator(lambda : dataset_producer(["val/br_z3_val/", "val/br_z4_val/",
                                                                            "val/pc_val/"],
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
    train_dataset = dataset_train.batch(4)
    validation_dataset = dataset_val.batch(1)
    test_dataset = dataset_test.batch(1)

    # Initialize and compile model
    generator = Generator_unet()
    LAMBDA = 100
    discriminator = Discriminator()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Save model : create a callback that saves the model's weights
    path_log_directory = os.path.join(log_directory, log_name)
    checkpoint_prefix = os.path.join(path_log_directory, "model_weights/cp.ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Restore weights from a pre-trained model (optional)
    #checkpoint.restore(tf.train.latest_checkpoint(os.path.join("/mnt/data3/mphilbert/output/log/20201118161329_pix2pix_loss/",
    #                                                      "model_weights/")))

    # Create a callback for visualization on tensorboard
    log_dir = os.path.join(path_log_directory, "visual/fit/")
    summary_writer = tf.summary.create_file_writer(log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Fit model
    EPOCHS = 150
    fit(train_dataset, EPOCHS)

    # Evaluate the model : print the mean Pearson correlation coefficient
    print("Pearson coeff:")
    print("\r train: {0:.5f}".format(pearson_coeff_batch(train_dataset)))
    print("\r validation: {0:.5f}".format(pearson_coeff(validation_dataset)))
    print("\r test: {0:.5f}".format(pearson_coeff(test_dataset)))

    # Run the trained model on a few examples from the test dataset and plot the generated images in a file named output_images
    os.mkdir(os.path.join(path_log_directory, "output_images/"))
    i = 0
    for inp, tar in test_dataset:
        generate_images(generator, inp, tar)
        plt.tight_layout()
        plt.savefig(os.path.join(path_log_directory, "output_images/") + "fig" + str(i) + ".png")
        i += 1

    end_script(start_time)
