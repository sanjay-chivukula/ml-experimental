import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

c_diff_kernel = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)

r_diff_kernel = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]
)


def immo_signal_viualization(file_path="data/train.mp4"):
    # matplotlib inits\
    plt.set_cmap("gray")
    r_immo_ax = plt.subplot(2, 2, 1)
    img_ax = plt.subplot(2, 2, 2)
    c_immo_ax = plt.subplot(2, 2, 4)

    video_obj = cv2.VideoCapture(file_path)

    if not video_obj.isOpened():
        print("Error opening file")
        exit(1)

    ret, prev_frame = video_obj.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    r_immo_prev, c_immo_prev = immo_transform(prev_frame)

    r_norm = preprocessing.Normalization()
    c_norm = preprocessing.Normalization()

    while video_obj.isOpened():
        ret, curr_frame = video_obj.read()
        # plt.clf()
        if ret:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            r_immo_curr, c_immo_curr = immo_transform(curr_frame)

            # do processing / training
            r_immo = np.concatenate((r_immo_prev, r_immo_curr),
                                    axis=0).reshape((1, -1), order='F')
            c_immo = np.concatenate((c_immo_prev, c_immo_curr),
                                    axis=0).reshape((1, -1), order='F')

            print(r_immo.shape, c_immo.shape)
            # Actual training process

            r_norm.adapt(r_immo)
            c_norm.adapt(c_immo)

            train(r_immo, c_immo, speed)

            # swapping prev with current data
            prev_frame, r_immo_prev, c_immo_prev = curr_frame, r_immo_curr, c_immo_curr
        else:
            continue

    plt.show()
    video_obj.release()


def immo_transform(frame):
    blur_kernel_size = (5, 5)
    blur_frame = cv2.GaussianBlur(frame, blur_kernel_size, 0)

    r_diff_frame = cv2.filter2D(blur_frame, -1, r_diff_kernel)
    c_diff_frame = cv2.filter2D(blur_frame, -1, c_diff_kernel)

    r_diff_frame = np.abs(np.sum(r_diff_frame, axis=1).reshape((1, -1)))
    c_diff_frame = np.abs(np.sum(c_diff_frame, axis=0).reshape((1, -1)))

    # these values are not normalized and is expected by model to normalize.
    return r_diff_frame, c_diff_frame


def train(r_data, c_data, speed_target):
    pass


def build_and_compile_model_r(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


if __name__ == "__main__":
    immo_signal_viualization()
