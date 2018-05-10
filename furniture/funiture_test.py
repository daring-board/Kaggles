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
from keras.models import load_model

def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

def join_fn_test(dat):
    return [dat['image_id'], dat['url'][0]]

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

    model_file_name = "funiture_cnn.h5"
#    for idx in range(250):
        # f = random.choice(f_list)
    offset = 200
    model = load_model(model_file_name)
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

        pred_class = model.predict(datas)
        for idx in range(len(f_list[itr*offset: (itr+1)*offset])):
            num = int(f_list[idx+itr*offset][5:-4])
            result[num] = np.argmax(pred_class, axis=1)[idx]
        elapse = time.time() - start
        print(elapse)

    with open('result.csv', 'w') as f:
        f.write('id,predicted\n')
        for key in result.keys():
            f.write('%d,%d\n'%(key, result[key]))
