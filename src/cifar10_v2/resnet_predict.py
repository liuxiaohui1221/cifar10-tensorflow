from tensorflow import keras
import cifar10_v2.LocalImport as local
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#save model
model_path='my_model.h5'

#超参数
# 残差块数
n = 3
depth = 6*n+2
batch_size = 128
epochs = 100
model=tf.keras.models.load_model(model_path)
#编译模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()

#model.save(model_path)
(x_train,y_train),(x_test,y_test) = local.get_files('C:/Downloads/cifar10/cleandata/clean','clean_label.txt')
print(x_train.shape,x_test.shape)
#计算类别数
num_labels = len(np.unique(y_train))
print("y_train: ",y_train)
#转化为one-hot编码
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print("one-hot y_train: ",y_train,y_train.shape)
#y_train=tf.squeeze(y_train)
print("one-hot squeeze y_train: ",y_train,y_train.shape)

#预处理
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

scores = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
# img_path = 'C:/Downloads/cifar10/cleandata/clean/43071.png'
# img = image.load_img(img_path, target_size=(32, 32))
# x = image.img_to_array(img)
# #print(x_test[0].shape)
# x = np.expand_dims(x_test[0], axis=0)
# x = preprocess_input(x)

results=model.predict(x_test,batch_size=batch_size,verbose=1)
max_index = np.argmax(results, axis=-1)
#print('Test results: ',scores)
print('Predict & actual results: ',max_index,y_test)
#print('Actual results: ',y_test)