import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import glob
import tqdm
import SimpleITK as sitk

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

from VQVAE import VQVAETrainer


##### ChestXray14 ######################################################################################################

data_root = '/home/user/data/prior_train/'

# train_val_df = pd.read_csv(os.path.join(data_root, 'train_val_list.csv'))
# train_val_df = train_val_df[train_val_df['No Finding'] == 1]
# train_images_df = data_root + '/images/' + train_val_df['Image Index']
train_images_df = list(map(lambda x: data_root + 'gt/' + x, os.listdir(data_root + 'gt/')))

a=1


def parse_function(filename):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize image with padding to 244x244
    image = tf.image.resize_with_pad(image, 256, 256, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)

    return image

def get_dataset(images_df, batch_size=32):

    dataset = tf.data.Dataset.from_tensor_slices((images_df))
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


########################################################################################################################


x_train = get_dataset(train_images_df)

# checkpoint_filepath = 'checkpoints/vqvae2'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints_healthy/vqvae2_healthy.ckpt", monitor='loss', verbose=1,
    # save_best_only=True,
    save_weights_only=True,
    mode='auto', save_freq='epoch')

# # Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# 
# # Open a strategy scope.
# with strategy.scope():
#     # Everything that creates variables should be under the strategy scope.
#     # In general this is only model construction & `compile()`.
vqvae_trainer = VQVAETrainer(1., latent_dim=32, num_embeddings=1024)
vqvae_trainer.compile(optimizer=keras.optimizers.Adam(1e-4))
vqvae_trainer.fit(x_train, epochs=100, callbacks=[checkpoint])

exit(0)


