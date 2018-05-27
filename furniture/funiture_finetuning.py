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

from keras.utils import np_utils, Sequence
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
    def __init__(self, num_classes, base_model):
        self.num_classes = num_classes
        self.shape = (128, 128, 3)
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

class DataSequence(Sequence):
    def __init__(self, kind, length, data_path):
        self.kind = kind
        self.length = length
        self.data_file_path = data_path
        # self.datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)

    def __getitem__(self, idx):
        f_list = os.listdir(self.data_file_path)
        f_list.sort()

        warp = 100
        aug_time = 2
        datas, labels = [], []

        for f in random.sample(f_list[idx:], warp):
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
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

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

    # モデル構築
    ft = FineTuning(129, model)
    model = ft.createNetwork()
    opt = ft.getOptimizer()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint('./checkpoints/weights.%02d.hdf5', verbose=1, save_weights_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'),
        ReduceLROnPlateau(factor=0.1, patience=1, verbose=1, cooldown=5, min_lr=1e-10),
        LambdaCallback(on_batch_begin=lambda batch, logs: print(' now: ',   datetime.datetime.now()))
    ]

    # fit model
    # model.fit(datas, labels, batch_size=50, epochs=n_epoch, callbacks=callbacks, validation_split=0.1)
    step_size = 100
    file_all = 50000
    train_gen = DataSequence('train', file_all, base_path)
    validate_gen = DataSequence('validate', 0.01*file_all, base_path)
    model.fit_generator(
        train_gen,
        steps_per_epoch=file_all/step_size,
        epochs=20,
        validation_data=validate_gen,
        validation_steps=0.01*file_all/step_size,
        )

    # save model
    model.save(model_file_name)
