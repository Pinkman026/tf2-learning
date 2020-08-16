import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.nn.softmax(x)
        return x
