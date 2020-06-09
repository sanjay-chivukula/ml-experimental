import numpy as np
import tensorflow as tf
from tensorflow import keras


def mnist_fashion_classifier():
    with tf.device("/CPU:0"):
        classifier_obj = MnistFashionClassifier()
        classifier_obj.init_model()
        classifier_obj.init_dataset()
        classifier_obj.train()
        print(classifier_obj.predict())


class MnistFashionClassifier:
    MNIST_IMAGE_WIDTH = 28
    MNIST_IMAGE_HEIGHT = 28
    MNIST_CLASSES = 10

    def __init__(self):
        self.model = self.init_model()
        self.train_dataset, self.test_dataset = self.init_dataset()

    def train(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        self.model.fit(
            self.train_dataset.repeat(),
            epochs=10,
            steps_per_epoch=500,
        )

    def predict(self):
        return np.argmax(self.model.predict(self.test_dataset)[0])

    @staticmethod
    def init_model():
        return keras.Sequential([
            keras.layers.Reshape(
                target_shape=(MnistFashionClassifier.MNIST_IMAGE_WIDTH * MnistFashionClassifier.MNIST_IMAGE_HEIGHT,),
                input_shape=(MnistFashionClassifier.MNIST_IMAGE_WIDTH, MnistFashionClassifier.MNIST_IMAGE_HEIGHT)
            ),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=192, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=10, activation='softmax'),
        ])

    def init_dataset(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        return self._create_dataset(x_train, y_train), self._create_dataset(x_test, y_test)

    @staticmethod
    def _preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.int64)
        return x, y

    def _create_dataset(self, xs, ys, n_classes=MNIST_CLASSES):
        ys = tf.one_hot(ys, depth=n_classes)
        return tf.data.Dataset.from_tensor_slices((xs, ys)) \
            .map(self._preprocess) \
            .shuffle(len(ys)) \
            .batch(128)


if __name__ == '__main__':
    mnist_fashion_classifier()
