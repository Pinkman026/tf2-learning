import tensorflow as tf
import numpy as np


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

