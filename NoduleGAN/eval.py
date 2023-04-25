import cv2
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

from gaugan import GauGAN


data_root = '/mnt/nas_houbb/users/Benjamin/LUNA16'

a=1


# 109198: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031_99.png'
# 109199: 'images/1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273_100.png'

# images = sorted(glob(os.path.join(data_root, f'images/*.png')))
# labels = sorted(glob(os.path.join(data_root, f'labels/*.png')))
#
# train_images, val_images = images[:109199], images[109199:]
# train_labels, val_labels = labels[:109199], labels[109199:]


# 1001: 'nodule_images/1.3.6.1.4.1.14519.5.2.1.6279.6001.504845428620607044098514803031_88.png'
# 1002: 'nodule_images/1.3.6.1.4.1.14519.5.2.1.6279.6001.511347030803753100045216493273_65.png'

nodule_images = sorted(glob(os.path.join(data_root, 'nodule_images/*.png')))
nodule_labels = sorted(glob(os.path.join(data_root, 'nodule_labels/*.png')))

train_nodule_images, val_nodule_images = nodule_images[:1002], nodule_images[1002:]
train_nodule_labels, val_nodule_labels = nodule_labels[:1002], nodule_labels[1002:]

# train_idx = np.random.default_rng(seed=42).choice(len(train_images), size=len(train_images)//100, replace=False)
# val_idx = np.random.default_rng(seed=42).choice(len(val_images), size=len(val_images)//100, replace=False)
#
# train_images_subset = train_nodule_images + [train_images[i] for i in train_idx]
# train_labels_subset = train_nodule_labels + [train_labels[i] for i in train_idx]
# val_images_subset = val_nodule_images + [val_images[i] for i in val_idx]
# val_labels_subset = val_nodule_labels + [val_labels[i] for i in val_idx]


IMG_HEIGHT = 512
NUM_CLASSES = 3
LATENT_DIM = 512
BATCH_SIZE = 16
NUM_EPOCHS = 100

def parse_function(image_file, label_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, 3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    label = tf.io.read_file(label_file)
    label = tf.image.decode_png(label, 1)
    label = tf.one_hot(label[..., 0], 3)

    return label, image, label

gaugan = GauGAN(IMG_HEIGHT, NUM_CLASSES, BATCH_SIZE, LATENT_DIM)
gaugan.load_weights('/tmp/checkpoints/gaugan_512x512.ckpt')


a=1

val_dataset = tf.data.Dataset.from_tensor_slices((val_nodule_images, val_nodule_labels))
val_dataset = val_dataset.map(parse_function)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)


iterator = iter(val_dataset)
batch = next(iterator)

a=1

latent_vector = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0.0, stddev=2.0)
fake_image = gaugan.predict([latent_vector, batch[0]])

for idx in range(BATCH_SIZE):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(batch[0][idx])
    ax2.imshow(batch[1][idx])
    ax3.imshow(fake_image[idx])
    plt.show()

