import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',  # valid or same
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.pool1(self.conv1(inputs))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.nn.softmax(x)
        return x