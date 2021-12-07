from tensorflow import keras
import numpy as np
import cifar10_v2.resnetModule as rs
import cifar10_v2.LocalImport as local
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#save model
model_path='my_model.h5'
#加载数据
#(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()
(x_train,y_train),(x_test,y_test) = local.get_files('C:/Downloads/cifar10/cleandata','clean','clean_label.txt')
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

#超参数
# 残差块数
n = 3
depth = 6*n+2
batch_size = 128
epochs = 100
#model=tf.keras.models.load_model(model_path)
# if model:
# 	pass
# else:
model = rs.resnet(input_shape=input_shape, depth=depth)
#编译模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()

print(len(x_train),len(y_train))
model.fit(x_train,y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test,y_test),
        shuffle=True)
model.save(model_path)
scores = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
results=model.predict(x_test)
print('Test results: ',results)
print('Test loss: ',scores[0])
print('Test accuracy: ',scores[1])