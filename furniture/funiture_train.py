# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import requests
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.callbacks import TensorBoard

from keras.utils import np_utils
from keras.models import load_model

class CNN:
    '''
    CNNで学習を行う。
    '''
    def __init__(self, train_data, num_classes):
        self.num_classes = num_classes
        self.shape = train_data.shape[1:]

    '''
    Networkを定義する
    '''
    def createNetwork(self):
        model = Sequential()
        model.add(Conv2D(124, (5, 5), activation='relu',
            input_shape=self.shape))
        model.add(Conv2D(122, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='softmax'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        return model


if __name__=="__main__":
    train      = json.load(open("./data/train.json"))
    test       = json.load(open("./data/test.json"))
    validation = json.load(open("./data/validation.json"))

    def join_fn(dat):
        return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

    train_df = pd.DataFrame(list(map(join_fn, zip(train['images'], train['annotations']))), columns=['id', 'url', 'label'])
    valid_df = pd.DataFrame(list(map(join_fn, zip(train['images'], validation['annotations']))), columns=['id', 'url', 'label'])
    test_df  = pd.DataFrame(list(map(lambda x: x["url"],test["images"])),columns=["url"])
    print(train_df.head(5))

    base_path = './data/download/'
    f_list = os.listdir(base_path)
    f_list.sort()
    print(f_list[:10])

    model_file_name = "funiture_org_cnn.h5"
    warp = 5000
    for iter  in range(1):
        datas, labels = [], []
        for f in f_list[iter*warp: (iter+1)*warp]:
            img = cv2.imread(base_path+f)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            tmp_df = train_df[train_df.id == int(f[6:-4])]
            labels.append(tmp_df['label'].iat[0])

        datas = np.asarray(datas)
        labels = pd.DataFrame(labels)
        n_class = labels.nunique().iat[0]
        print(labels.head(5))
        labels = np_utils.to_categorical(labels, 129)
        print(datas.shape[1:])

        if iter == 0:
            # モデル構築
            cnn = CNN(datas, 129)
            model = cnn.createNetwork()
            adam = Adam(lr=1e-4)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
        else:
            model = load_model(model_file_name)
        # fit model
        model.fit(datas, labels, batch_size=10, epochs=30)
        # save model
        model.save(model_file_name)
