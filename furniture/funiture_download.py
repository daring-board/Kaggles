# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import requests
import numpy as np
import pandas as pd

train      = json.load(open("./data/train.json"))
test       = json.load(open("./data/test.json"))
validation = json.load(open("./data/validation.json"))

def join_fn(dat):
    return [dat[0]['image_id'], dat[0]['url'][0], dat[1]['label_id']]

train_df = pd.DataFrame(list(map(join_fn, zip(train['images'], train['annotations']))), columns=['id', 'url', 'label'])
valid_df = pd.DataFrame(list(map(join_fn, zip(train['images'], validation['annotations']))), columns=['id', 'url', 'label'])
test_df  = pd.DataFrame(list(map(lambda x: x["url"],test["images"])),columns=["url"])

for idx, row in train_df.sample(n=1000).iterrows():
    url = row['url']
    img_id = row['id']
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
        cv2.imwrite('./data/download/train_%09d.jpg'%img_id, img)
    except:
        continue
