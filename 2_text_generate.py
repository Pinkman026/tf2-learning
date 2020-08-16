import tensorflow as tf
import numpy as np
from data_loaders.text_loader import TextLoader
from models.rnn import RNN


def train():
    num_batches = 10000
    seq_length = 40
    batch_size = 50
    show = 100
    learning_rate = 1e-3

    data_loader = TextLoader()
    model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    running_loss = 0
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(seq_length=seq_length, batch_size=batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            running_loss += loss.numpy()

            if batch_index % show == 0:
                print("batch: {}, loss: {:.4f}".format(batch_index, running_loss / show))
                running_loss = 0
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return data_loader, model


def generate(data_loader, model):
    X_, _ = data_loader.get_batch(seq_length=40, batch_size=1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity: {}".format(diversity))
        for t in range(400):
            y_pred = model.predict(X, diversity)
            print(data_loader.indices_char[y_pred[0]], end="", flush=False)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print("\n")


if __name__ == '__main__':
    dl, m = train()
    generate(dl, m)
