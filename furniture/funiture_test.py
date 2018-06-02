# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import time
import random
import requests
import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import load_model, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential
from keras.utils import np_utils, Sequence


def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

def join_fn_test(dat):
    return [dat['image_id'], dat['url'][0]]

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
            # opt = SGD(lr=1e-4, momentum=0.9)
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

if __name__=="__main__":
    train      = json.load(open("./data/train.json"))
    test       = json.load(open("./data/test.json"))
    validation = json.load(open("./data/validation.json"))

    train_df = pd.DataFrame(list(map(join_fn, zip(train['images'], train['annotations']))), columns=['id', 'url', 'label'])
    valid_df = pd.DataFrame(list(map(join_fn, zip(train['images'], validation['annotations']))), columns=['id', 'url', 'label'])
    test_df  = pd.DataFrame(list(map(join_fn_test, test["images"])), columns=['id', 'url'])
    result = {int(row['id']): 0 for idx, row in test_df.iterrows()}

    base_path = './data/download_test/'
    f_list = os.listdir(base_path)
    f_list.sort()

    offset = 200
    ft = FineTuning(129, 'ResNet')
    model1 = ft.createNetwork()
    model1.load_weights('checkpoints/weights.10-0.09-0.98-0.88-0.84.hdf5')
    ft = FineTuning(129, 'VGG16')
    model2 = ft.createNetwork()
    model2.load_weights('checkpoints/weights_vgg16.hdf5')
    for itr in range(int(len(test_df.index)/offset)):
        start = time.time()
        datas, labels, files = [], [], []
        for f in f_list[itr*offset: (itr+1)*offset]:
            img = cv2.imread(base_path+f)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            files.append(base_path+f)

        cv2.destroyAllWindows()
        datas = np.asarray(datas)
        if len(datas) == 0: break

        pred_class1 = model1.predict(datas)
        pred_class2 = model2.predict(datas)

        for idx in range(len(f_list[itr*offset: (itr+1)*offset])):
            num = int(f_list[idx+itr*offset][5:-4])
            ems = pred_class1[idx] + pred_class2[idx]
            result[num] = np.argmax(ems)
        elapse = time.time() - start
        print(elapse)

    with open('result.csv', 'w') as f:
        f.write('id,predicted\n')
        for key in result.keys():
            f.write('%d,%d\n'%(key, result[key]))
