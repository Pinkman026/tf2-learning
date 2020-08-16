# tf2 学习笔记

结合官方教程使用：https://tf.wiki/zh_hans/mlstudyjam2nd.html

一些函数记录:

* tf.keras.layers.Flatten():   # Flatten层将除第一维（batch_size）展平
* with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
* tf.keras.layers.Dense(units, activation, use_bias, kernel_initializer
* tf.keras.losses.sparse_categorical_crossentropy :交叉熵损失函数
* categorical_crossentropy: one hot 交叉熵损失函数
* tf.keras.metrics.SparseCategoricalAccuracy 模型评估，batch的形式，最后 .result() 得到最终结果
* self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
	)
* tf.keras.applications. 下面有很多预训练的网络结构，包括Xception，ResNet，Inception，MobileNet，DenseNet，EfficientNet等，
详细参考：https://keras.io/api/applications/ ，变量主要有以下几个
    * input_shape: 	输入张量的形状（不含第一维的 Batch），大多默认为 224 × 224 × 3 。一般而言，模型对输入张量的大小有下限，长和宽至少为 32 × 32 或 75 × 75 ；
    * include_top ：在网络的最后是否包含全连接层，默认为 True ；
    * weights ：预训练权值，默认为 'imagenet' ，即为当前模型载入在 ImageNet 数据集上预训练的权值。如需随机初始化变量可设为 None ；
    * classes ：分类数，默认为 1000。修改该参数需要 include_top 参数为 True 且 weights 参数为 None 。
* tf.shape(inputs): 可以获得更多的信息
* 查看模型参数信息，先给模型一个输入，然后打印信息
    * model.build(input_shape=(None, 28 * 28))
    * model.summary()

## 在mnist上训练

使用 `MLP` 训练或者 `CNN` 训练

```bash
python train_mnist.py --model mlp
python train_mnist.py --model cnn
```

## RNN 文本生成

模型定义

```python
class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)  # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
        logits = self.dense(output)
        if from_logits:  # from_logits 参数控制输出是否通过 softmax 函数进行归一化
            return logits
        else:
            return tf.nn.softmax(logits)
    def predict(self, inputs, temeraure=1.):
        """
        按概率分布生成一个词，而不是取概率可能最大的词
        :param inputs: batch
        :param temeraure: 控制分布的形状，越大分布越平缓（最大值和最小值的差值越小），生成文本的丰富度越高；越小则分布越陡峭，生成文本的丰富度越低。
        :return: 对batch中每个数据生成一个词
        """

        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temeraure).numpy()
        select = np.array([
            np.random.choice(self.num_chars, p=prob[i, :]) for i in range(batch_size.numpy())
        ])
        return select
```

## 模型保存和加载

* checkpoint = tf.train.Checkpoint(myAwesomeModel=model) 实例化保存
* checkpoint.save('./save/model.ckpt') 保存
* checkpoint.restore(tf.train.latest_checkpoint('./save')) 读取
* tf.train.latest_checkpoint(save_path) 返回最新保存的模型路径
* checkpoint 也支持图执行模式
* 通过 tf.train.CheckpointManager 来定义保存属性
    * manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt', max_to_keep=k)
    * 使用CheckpointManager保存模型参数到文件并自定义编号
    * path = manager.save(checkpoint_number=batch_index)
    
myAwesomeModel 是待保存的模型 model 所取的任意键名。注意，在恢复变量的时候，还将使用这一键名。
代码示例：

```python
## 训练阶段
model = MyModel()
# 实例化Checkpoint，设置保存对象为model
checkpoint = tf.train.Checkpoint(myAwesomeModel=model)

if batch_index % 100 == 0:                              # 每隔100个Batch保存一次
	path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
	print("model saved to %s" % path)

## 测试阶段
model_to_be_restored = MyModel()
checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)      
checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
```

## TensorBoard 训练过程可视化

* 实例化writer：summary_writer = tf.summary.create_file_writer('./tensorboard')
* 开启写入环境：with summary_writer.as_default():
* 写入文件
    * 标量：tf.summary.scalar(name, tensor, step=batch_index)
    * 图像：tf.summary.image('Traning sample:', sample_img, step=0)  # sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
* 开启：tensorboard --logdir=./tensorboard

关键代码

```python
summary_writer = tf.summary.create_file_writer(log_dir)  # 实例化记录器
tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        with summary_writer.as_default():  # 指定记录器
            tf.summary.scalar("loss", loss, step=batch_index)  # 将当前损失函数的值写入记录器
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)  # 保存Trace信息到文件（可选）
```

## tf.data

* 基础用法：dataset = tf.data.Dataset.from_tensor_slices((X, Y))，X Y 可以是tensor，也可以是numpy数组
* 对于大数据集，可以处理成TFRecord格式，然后使用tf.data.TFRocordDataset()载入
* tf.data.Dataset 常用函数
    * map(f, num_parallel_calls=2) ：对数据集中的每个元素应用函数 f （函数输入是x和y，输出也是x和y），得到一个新的数据集（这部分往往结合 tf.io 进行读写和解码文件， tf.image 进行图像处理）；可以并行处理
    * shuffle(buffer_size) ：将数据集打乱
    * batch(batch_size)：将数据集分成批次，即对每 batch_size 个元素，使用 tf.stack() 在第 0 维合并，成为一个元素；
    * repeat()：重复数据中的元素
    * prefetch(buffer_size=tf.data.experimental.AUTOTUNE)：用法和shuffle类似，预先加载到内存，buffer_size可以自动设置

使用dataset搭配keras接口：

```python
model.fit(mnist_dataset, epochs=num_epochs)
model.evaluate(test_dataset)
```

## TFRecord

读取 image 数据，并写入 TFRecord ，关键代码：

```python
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
        feature = {  # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))  # 通过字典建立 Example
        writer.write(example.SerializeToString())  # 将Example序列化并写入 TFRecord 文件
```

读取 TFRecord：

```python
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
```