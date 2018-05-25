# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import datetime
import random
import requests
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ProgbarLogger, ReduceLROnPlateau, LambdaCallback
from keras.layers import Input

from keras.utils import np_utils
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator

class FineTuning:
    '''
    CNNで学習を行う。(転移学習)
    Avable base_model is
        VGG16, DenseNet201, ResNet50
    '''
    def __init__(self, train_data, num_classes, base_model):
        self.num_classes = num_classes
        self.shape = train_data.shape[1:]
        self.input_tensor = Input(shape=self.shape)
        self.base_model = base_model
        if base_model == 'VGG16':
            self.base = VGG16(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        elif base_model == 'DenseNet201':
            self.base = DenseNet201(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
        else:
            self.base = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=self.input_tensor)
            # self.base = ResNet50(include_top=False, weights='imagenet', input_tensor=self.input_tensor)

    def getOptimizer(self):
        if self.base_model == 'VGG16':
            #opt = SGD(lr=1e-4, momentum=0.9)
            opt = Adam(lr=1e-4)
        elif self.base_model == 'DenseNet201':
            opt = Adam(lr=1e-4)
        else:
            opt = Adam(lr=1e-4)
        return opt

    '''
    Networkを定義する
    '''
    def createNetwork(self):
        tmp_model = Sequential()
        tmp_model.add(Flatten(input_shape=self.base.output_shape[1:]))
        tmp_model.add(Dense(256, activation='relu'))
        tmp_model.add(Dropout(0.5))
        tmp_model.add(Dense(self.num_classes, activation='softmax'))

        model = Model(input=self.base.input, output=tmp_model(self.base.output))
        for layer in model.layers[:15]:
            layer.trainable = False
        return model

def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

if __name__=="__main__":
    model = sys.argv[1]

    train      = json.load(open("./data/train.json"))
    test       = json.load(open("./data/test.json"))
    validation = json.load(open("./data/validation.json"))

    train_df = pd.DataFrame(list(map(join_fn, zip(train['images'], train['annotations']))), columns=['id', 'url', 'label'])
    valid_df = pd.DataFrame(list(map(join_fn, zip(train['images'], validation['annotations']))), columns=['id', 'url', 'label'])
    test_df  = pd.DataFrame(list(map(lambda x: x["url"],test["images"])),columns=["url"])
    print(train_df.head(5))

    base_path = './data/download/'
    f_list = os.listdir(base_path)
    f_list.sort()
    print(f_list[:10])

    model_file_name = "funiture_cnn.h5"
    datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)

    warp = 10000
    aug_time = 2
    n_epoch = 100
    datas, labels = [], []

    for f in random.sample(f_list, warp):
        img = cv2.imread(base_path+f)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
        tmp_df = train_df[train_df.id == int(f[6:-4])]
        labels.append(tmp_df['label'].iat[0])
        # Augmentation image
        # for num in range(aug_time):
        #     tmp = datagen.random_transform(img)
        #     datas.append(tmp)
        #     labels.append(tmp_df['label'].iat[0])

    datas = np.asarray(datas)
    labels = pd.DataFrame(labels)
    labels = np_utils.to_categorical(labels, 129)

    for iter in range(4):
        if iter == 0:
            # モデル構築
            ft = FineTuning(datas, 129, model)
            model = ft.createNetwork()
            opt = ft.getOptimizer()
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
        # if iter == 0:
        #     model = load_model(model_file_name)
        #     wgt = './checkpoints_vgg16_early/weights.12-00.hdf5'
        #     model.load_weights(wgt)
        else:
            model = load_model(model_file_name)
            for num in reversed(range(n_epoch)):
                wgt = './checkpoints/weights.%02d-%02d.hdf5'%(num, iter)
                if not os.path.isfile(wgt): continue
                model.load_weights(wgt)

        callbacks = [
            ModelCheckpoint('./checkpoints/weights.{epoch:02d}-%02d.hdf5'%iter, verbose=1, save_weights_only=True, monitor='val_loss'),
            EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=3, verbose=0, mode='auto'),
            ReduceLROnPlateau(factor=0.05, patience=1, verbose=1),
            LambdaCallback(on_batch_begin=lambda batch, logs: print(' now: ',   datetime.datetime.now()))
        ]

        # fit model
        model.fit(datas, labels, batch_size=50, epochs=n_epoch, callbacks=callbacks, validation_split=0.1)
        # save model
        model.save(model_file_name)
