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

    model_file_name = "funiture_cnn.h5"
    datas, labels, files = [], [], []
    for idx in range(250):
        f = random.choice(f_list)
        img = cv2.imread(base_path+f)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
        files.append(base_path+f)
        tmp_df = train_df[train_df.id == int(f[6:-4])]
        labels.append(tmp_df['label'].iat[0])

    datas = np.asarray(datas)
    labels_org = pd.DataFrame(labels)
#    for row in labels.iterrows(): print(row)
    n_class = labels_org.nunique().iat[0]
    labels = np_utils.to_categorical(labels_org, 129)
    model = load_model(model_file_name)

    # evaluate model
    score = model.evaluate(datas, labels, verbose=0)
    print('test loss:', score[0])
    print('test acc:', score[1])

    pred_size = 5
    pred_class = model.predict(datas[-pred_size:])
    for idx in reversed(range(1, pred_size+1)):
        f = files[-idx]
        img = cv2.imread(f)
        cv2.imshow(f[len(base_path)+6:-4], img)
        print('%s: %d'%(f, labels_org.iloc[-idx, 0]))
    print(np.argmax(pred_class, axis=1))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for idx in range(pred_size):
    #     print('correct: %d, predct: %d'%(labels[idx], pred_class[idx]))
