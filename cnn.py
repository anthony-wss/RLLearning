from argparse import ArgumentParser
from email.policy import default
from multiprocessing import pool
from keras.datasets import cifar10
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pickle


def main(args):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    if args.do_train:

        # 0. Preview Training data
        """
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(train_images[i])
        plt.show()
        """

        # 1. Preprocess

        # Min-Max Normalization
        train_images = train_images.astype("float32") / 255.0
        # One-hot Encoding
        train_labels = to_categorical(train_labels, 10)

        # 2. Building Model

        # First Convolution Block:
        # Conv -> Conv -> Pool -> Dropout
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=train_images[0].shape,
            )
        )
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(args.dropout_rate))

        # Second Convolution Block:
        # Conv -> Conv -> Pool -> Dropout
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(args.dropout_rate))

        # Flatten -> Dense -> Dropout -> Dense
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(args.dropout_rate))
        model.add(Dense(10, activation="softmax"))

        # 3. Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["acc"],
        )

        # 4. Train the model
        history = model.fit(
            train_images, train_labels, batch_size=128, epochs=20, validation_split=0.1
        )
        pickle.dump(history.history, open(args.results_path, "wb"))


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="cnn.pkl",
        help="path to model.pkl file, default is ./cnn.pkl",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Set true to run traing process"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="cnn_results.pkl",
        help="path to results.pkl file, default is ./cnn_results.pkl",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.25, help="dropout rate, default is 0.25"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate, default is 0.001"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    main(args)
