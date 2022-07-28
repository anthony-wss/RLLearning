from keras.datasets import boston_housing

from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pandas as pd
import numpy as np
import pickle
from argparse import ArgumentParser

def train_model(args):
    (train_data, train_labels), _ = boston_housing.load_data()

    # column_names = ['CRIM', "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    # df = pd.DataFrame(train_data, columns=column_names)
    # print(df.head())

    # Shuffle
    order = np.random.randint(0, 404, size=404)
    train_data = train_data[order]
    train_labels = train_labels[order]

    # axis=0 為鉛直方向
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit(train_data, train_labels, batch_size=32, epochs=500, validation_split=0.2, callbacks=[early_stop])
    # history = model.fit(train_data, train_labels, batch_size=32, epochs=500, validation_split=0.2)
    
    pickle.dump(history.history, open("results.pkl", "wb"))
    pickle.dump(model, open(args.model_path, "wb"))
    return model

def main(args):

    if args.do_train:
        model = train_model(args)
    else:
        model = pickle.load(open(args.model_path, "rb"))
    
    if args.do_pred:
        _, (test_data, test_labels) = boston_housing.load_data()
    
        mean = test_data.mean(axis=0)
        std = test_data.std(axis=0)
        test_data = (test_data - mean) / std

        test_loss, test_mae = model.evaluate(test_data, test_labels)

        print(f"testing loss: {test_loss}")
        print(f"test mae: {test_mae}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True, help="path to model.pkl file")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_pred", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)