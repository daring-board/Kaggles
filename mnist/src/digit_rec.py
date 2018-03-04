import time
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import tensorflow as tf
import pandas as pd
from sklearn.datasets import fetch_mldata

# ネットワークの定義
# プレースホルダー
x_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
# 畳み込み層1
conv1_features = 20 # 畳み込み層1の出力次元数
max_pool_size1 = 2 # 畳み込み層1のマックスプーリングサイズ
conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, conv1_features], stddev=0.1), dtype=tf.float32) # 畳み込み層1の重み
conv1_b = tf.Variable(tf.constant(0.1, shape=[conv1_features]), dtype=tf.float32) # 畳み込み層1のバイアス
conv1_c2 = tf.nn.conv2d(x_, conv1_w, strides=[1, 1, 1, 1], padding="SAME") # 畳み込み層1-畳み込み
conv1_relu = tf.nn.relu(conv1_c2+conv1_b) # 畳み込み層1-ReLU
conv1_mp = tf.nn.max_pool(conv1_relu, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding="SAME") # 畳み込み層1-マックスプーリング
# 畳み込み層2
conv2_features = 50 # 畳み込み層2の出力次元数
max_pool_size2 = 2 # 畳み込み層2のマックスプーリングのサイズ
conv2_w = tf.Variable(tf.truncated_normal([5, 5, conv1_features, conv2_features], stddev=0.1), dtype=tf.float32) # 畳み込み層2の重み
conv2_b = tf.Variable(tf.constant(0.1, shape=[conv2_features]), dtype=tf.float32) # 畳み込み層2のバイアス
conv2_c2 = tf.nn.conv2d(conv1_mp, conv2_w, strides=[1, 1, 1, 1], padding="SAME") # 畳み込み層2-畳み込み
conv2_relu = tf.nn.relu(conv2_c2+conv2_b) # 畳み込み層2-ReLU
conv2_mp = tf.nn.max_pool(conv2_relu, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding="SAME") # 畳み込み層2-マックスプーリング
# 全結合層1
result_w = x_.shape[1] // (max_pool_size1*max_pool_size2)
result_h = x_.shape[2] // (max_pool_size1*max_pool_size2)
fc_input_size = result_w * result_h * conv2_features # 畳み込んだ結果、全結合層に入力する次元数
fc_features = 500 # 全結合層の出力次元数（隠れ層の次元数）
s = conv2_mp.get_shape().as_list() # [None, result_w, result_h, conv2_features]
conv_result = tf.reshape(conv2_mp, [-1, s[1]*s[2]*s[3]]) # 畳み込みの結果を1*N層に変換
fc1_w = tf.Variable(tf.truncated_normal([fc_input_size.value, fc_features], stddev=0.1), dtype=tf.float32) # 重み
fc1_b = tf.Variable(tf.constant(0.1, shape=[fc_features]), dtype=tf.float32) # バイアス
fc1 = tf.nn.relu(tf.matmul(conv_result, fc1_w)+fc1_b) # 全結合層1
# 全結合層2
fc2_w = tf.Variable(tf.truncated_normal([fc_features, fc_features], stddev=0.1), dtype=tf.float32) # 重み
fc2_b = tf.Variable(tf.constant(0.1, shape=[fc_features]), dtype=tf.float32) # バイアス
fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w)+fc2_b) # 全結合層2
# 全結合層3
fc3_w = tf.Variable(tf.truncated_normal([fc_features, 10], stddev=0.1), dtype=tf.float32) # 重み
fc3_b = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32) # バイアス
y = tf.matmul(fc2, fc3_w)+fc3_b
# クロスエントロピー誤差
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 勾配法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 正解率の計算
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 学習
EPOCH_NUM = 30
BATCH_SIZE = 1000

# 教師データ
train_df = pd.read_csv('../input/train.csv')
train_x, train_y = train_df.ix[:,1:train_df.shape[1]].as_matrix(), train_df.ix[:,0].as_matrix()
train_x = train_x / 255 # 0-1に正規化する

# 教師データを変換
train_x = train_x.reshape([-1, 28, 28, 1]) # (N, height, width, channel)
# ラベルはone-hotベクトルに変換する
train_y = np.eye(np.max(train_y)+1)[train_y]

saver = tf.train.Saver()
# 学習
print("Train")
with tf.Session() as sess:
    st = time.time()
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH_NUM):
        total_loss = 0
        for i in range(0, train_x.shape[0], BATCH_SIZE):
            batch_x = train_x[i: i+BATCH_SIZE]
            batch_y = train_y[i: i+BATCH_SIZE]
            total_loss += cross_entropy.eval(feed_dict={x_: batch_x, y_: batch_y})
            train_step.run(feed_dict={x_: batch_x, y_: batch_y})
        test_accuracy = accuracy.eval(feed_dict={x_: batch_x, y_: batch_y})
        if (epoch+1) % 1 == 0:
            ed = time.time()
            print("epoch:\t{}\ttotal loss:\t{}\tvaridation accuracy:\t{}\ttime:\t{}".format(epoch+1, total_loss, test_accuracy, ed-st))
            st = time.time()

    # モデルの保存
    saver.save(sess, "./model.ckpt")
    # モデルのリストア
    # saver.restore(sess, "model.ckpt")

    # 予測データ
    test_df = pd.read_csv('../input/test.csv')
    test_x = test_df.as_matrix()
    test_x = test_x / 255 # 0-1に正規化する

    # 予測データを変換
    test_x = test_x.reshape([-1, 28, 28, 1]) # (N, height, width, channel)
    with open('submit.csv', 'w') as f:
        f.write('ImageId,Label\n')
        for i in range(0, test_x.shape[0]):
            pre = np.argmax(y.eval(feed_dict={x_: [test_x[i]]}))
            f.write('%d,%d\n'%(i+1, pre))
