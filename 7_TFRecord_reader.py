import tensorflow as tf
import matplotlib.pyplot as plt

data_dir = 'data/fastai-datasets-cats-vs-dogs-2'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
tfrecord_file = data_dir + '/train/train.tfrecords'

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件

feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码JPEG图片
    return feature_dict['image'], feature_dict['label']


dataset = raw_dataset.map(_parse_example)

for image, label in dataset:
    plt.title('cat' if label == 0 else 'dog')
    plt.imshow(image.numpy())
    plt.show()
