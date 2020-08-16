import tensorflow as tf
import numpy as np
from utils.mnist_loader import MNISTLoader
from models.mlp import MLP
from models.cnn import CNN
import argparse


def train(name):
    name = name.upper()
    assert name in ["MLP", "CNN"]
    num_steps = 5
    batch_size = 20
    learning_rate = 0.001
    show_every = 100

    if name == "MLP":
        model = MLP()
    else:
        model = CNN()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    num_batches = num_steps * data_loader.num_train_data // batch_size
    running_loss = 0
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        if batch_index % show_every == 0:
            print("batch {}, loss: {:.4f}".format(batch_index, running_loss / show_every))
            running_loss = 0
        running_loss += loss.numpy()
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # 效果评估
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MLP", type=str)
    args = parser.parse_args()
    train(args.model)
