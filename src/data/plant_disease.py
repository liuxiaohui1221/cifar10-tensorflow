# Install nightly package for some functionalities that aren't in alpha
#!pip install tensorflow-gpu==2.0.0-beta1
# Install TF Hub for TF2
#!pip install 'tensorflow-hub == 0.5'

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

#Load data
zip_file=tf.keras.utils.get_file(origin='https://storage.googleapis.com/plantdata/PlantVillage.zip',
 fname='PlantVillage.zip', extract=True)
#Create the training and validation directories
data_dir = os.path.join(os.path.dirname(zip_file), 'PlantVillage')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

#!wget https: // github.com / obeshor / Plant - Diseases - Detector / archive / master.zip
#!unzip master.zip;
import json

with open('backups/plant_disease/Plant-Diseases-Detector-master/categories.json', 'r') as f:
 cat_to_name = json.load(f)
 classes = list(cat_to_name.values())

print(classes)


module_selection = ("inception_v3", 299, 2048) #@param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/tf2- preview/{}/feature_vector/2".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 64 #@param {type:"integer"}

# Inputs are suitably resized for the selected module.
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
 validation_dir,
 shuffle=False,
 seed=42,
 color_mode="rgb",
 class_mode="categorical",
 target_size=IMAGE_SIZE,
 batch_size=BATCH_SIZE)
do_data_augmentation = True  # @param {type:"boolean"}
if do_data_augmentation:
 train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale=1. / 255,
  rotation_range=40,
  horizontal_flip=True,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  fill_mode='nearest')
else:
 train_datagen = validation_datagen

train_generator = train_datagen.flow_from_directory(
 train_dir,
 subset="training",
 shuffle=True,
 seed=42,
 color_mode="rgb",
 class_mode="categorical",
 target_size=IMAGE_SIZE,
 batch_size=BATCH_SIZE)