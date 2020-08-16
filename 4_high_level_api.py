import tensorflow as tf
import argparse
from data_loaders.mnist_loader import MNISTLoader


def construct_model(how="sequential"):
    if how == "sequential":
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Softmax()
        ])
    elif how == "functional":
        # 帮助建立更为复杂的模型，例如多输入 / 输出或存在参数共享的模型。其使用方法是将层作为可调用的对象并返回张量，
        # 并将输入向量和输出向量提供给 tf.keras.Model 的 inputs 和 outputs 参数
        inputs = tf.keras.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(units=10)(x)
        outputs = tf.keras.layers.Softmax()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main(how):
    num_epochs = 5
    batch_size = 20
    model = construct_model(how=how)
    data_loader = MNISTLoader()

    # 配置训练过程
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

    # 评估
    print(model.evaluate(data_loader.test_data, data_loader.test_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="sequential", type=str)
    args = parser.parse_args()

    main(how=args.api)
