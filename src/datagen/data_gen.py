import tensorflow as tf


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
  rescale=1. / 255,
  rotation_range=40,
  horizontal_flip=True,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  fill_mode='nearest')


IMAGE_SIZE = (299, 299)
BATCH_SIZE = 64
train_dir='C:/Downloads/data/train'
save_dir='C:/Downloads/datagen/train'
train_generator = train_datagen.flow_from_directory(
 train_dir,
 save_to_dir=save_dir,
 save_prefix='gen',
 subset="training",
 shuffle=True,
 seed=42,
 color_mode="rgb",
 class_mode="categorical",
 target_size=IMAGE_SIZE,
 batch_size=BATCH_SIZE)

train_datagen.