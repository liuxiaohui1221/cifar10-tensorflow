# -*- coding: utf8 -*-
# author: ronniecao
import os
import src.data.plant_disease as pl
import tensorflow._api.v2.compat.v1 as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cifar10 = pl.Plant()

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=38, image_size=24, network_path='src/config/networks/basic.yaml')
    #convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/plant-v1/', batch_size=128, n_epoch=500)
    convnet.test(dataloader=cifar10, backup_path='backups/plant-v2/', epoch=5000, batch_size=128)
    convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def vgg_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=38, image_size=24, network_path='src/config/networks/vgg.yaml')
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/plant-v2/', batch_size=128, n_epoch=500)
    # convnet.test(backup_path='backups/cifar10-v3/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def resnet():
    from src.model.resnet import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=38, image_size=24, network_path='src/config/networks/resnet.yaml')
    convnet.train(dataloader=cifar10, backup_path='backups/plant-v5/', batch_size=128, n_epoch=500)
    convnet.test(backup_path='backups/plant-v4/', epoch=0, batch_size=128)

#basic_cnn()
#vgg_cnn()
resnet()
