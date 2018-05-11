# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import requests
import numpy as np
import pandas as pd

mode = sys.argv[1]

train      = json.load(open("./data/train.json"))
test       = json.load(open("./data/test.json"))
validation = json.load(open("./data/validation.json"))

def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

def join_fn_test(dat):
    return [dat['image_id'], dat['url'][0]]

train_df = pd.DataFrame(list(map(join_fn, zip(train['images'], train['annotations']))), columns=['id', 'url', 'label'])
valid_df = pd.DataFrame(list(map(join_fn, zip(validation['images'], validation['annotations']))), columns=['id', 'url', 'label'])
test_df  = pd.DataFrame(list(map(join_fn_test, test["images"])), columns=['id', 'url'])

for idx, row in train_df.sample(n=1000).iterrows():
#for idx, row in test_df.iterrows():
    url = row['url']
    img_id = row['id']
#    path = './data/download_test/%s_%09d.jpg'%(mode, img_id)
    path = './data/download/%s_%09d.jpg'%(mode, img_id)
    if os.path.isfile(path): continue
    print(row)

    try:
        res = requests.get(url, stream=True, timeout=15)
        tmp_file = './data/tmp.jpg'
        with open(tmp_file, 'wb') as f:
            f.write(res.content)

        img = cv2.imread(tmp_file)
        print(img.shape)
        img = cv2.resize(img, (256, 256))
        print(img.shape)
#        cv2.imwrite('./data/download_test/%s_%09d.jpg'%(mode, img_id), img)
        cv2.imwrite('./data/download/%s_%09d.jpg'%(mode, img_id), img)
    except Exception as e:
        print(e.args)
        continue
