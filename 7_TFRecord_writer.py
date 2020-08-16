import tensorflow as tf
import os

data_dir = 'data/fastai-datasets-cats-vs-dogs-2'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_file = data_dir + '/train/train.tfrecords'

train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
# 各写入5个
train_filenames = train_cat_filenames[:5] + train_dog_filenames[:5]
# 将 cat 类的标签设为0，dog 类的标签设为1
train_labels = [0] * 5 + [1] * 5

with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {  # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
        writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
