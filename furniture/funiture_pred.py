# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import random
import requests
import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

if __name__=="__main__":
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

    datas, labels, files = [], [], []
    datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.5)
    aug_time = 4
    for idx in range(250):
        f = random.choice(f_list)
        img = cv2.imread(base_path+f)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        tmp_df = train_df[train_df.id == int(f[6:-4])]
        datas.append(img)
        files.append(base_path+f)
        labels.append(tmp_df['label'].iat[0])
        # Augmentation image
        for num in range(aug_time):
            tmp = datagen.random_transform(img)
            datas.append(tmp)
            files.append(base_path+f)
            labels.append(tmp_df['label'].iat[0])


    datas = np.asarray(datas)
    labels_org = pd.DataFrame(labels)
#    for row in labels.iterrows(): print(row)
    n_class = labels_org.nunique().iat[0]
    labels = np_utils.to_categorical(labels_org, 129)

    model_file_name = "funiture_cnn_resnet.h5"
    model = load_model(model_file_name)
    # evaluate model
    # score = model.evaluate(datas, labels, verbose=0)
    # print('test loss:', score[0])
    # print('test acc:', score[1])
    pred_class1 = model.predict(datas)

    model_file_name = "funiture_cnn_vgg16_early.h5"
    model = load_model(model_file_name)
    model.load_weights('checkpoints_vgg16_early/weights.07-00.hdf5')
    # evaluate model
    # score = model.evaluate(datas, labels, verbose=0)
    # print('test loss:', score[0])
    # print('test acc:', score[1])
    pred_class2 = model.predict(datas)

    with open('validate.csv', 'w') as f:
        for idx in range(0, len(datas)+1, aug_time+1):
            f_name = files[idx]
            label = labels_org.iloc[idx, 0]
            # img = cv2.imread(f)
            # cv2.imshow(f[len(base_path)+6:-4], img)
            #print('%s: %d, %s'%(f_name, label, str(pred_class[idx][label])))
            tta_val1 = np.sum(pred_class1[idx: idx+aug_time+1], axis=0) / aug_time+1
            tta_val2 = np.sum(pred_class2[idx: idx+aug_time+1], axis=0) / aug_time+1
            ems = (pred_class1[idx] + pred_class2[idx])/2
            tta_ems = (tta_val1 + tta_val2)/2
            all_ems = (ems + tta_ems)/2
            # f.write('%s: %d, %d, %s\n'%(f_name, label, np.argmax(pred_class[idx]), np.argmax(tta_val)))
            f.write('%s: %d, %d, %d, '%(f_name, label, np.argmax(pred_class1[idx]), np.argmax(pred_class2[idx])))
            f.write('%d, %d, %d\n'%(np.argmax(ems), np.argmax(tta_ems), np.argmax(all_ems)))
        # print(np.argmax(pred_class, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for idx in range(pred_size):
    #     print('correct: %d, predct: %d'%(labels[idx], pred_class[idx]))
