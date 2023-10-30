import os
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging as log
import numpy as np


def shuffle_np(x, y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def shuffle_and_undersample(x, y):
    x, y = shuffle_np(x, y)

    min_len = np.minimum(np.sum(y == 1), np.sum(y == 0))

    x_zeros = x[(y == 0).reshape(-1)][:min_len]
    x_ones = x[(y == 1).reshape(-1)][:min_len]

    x = np.concatenate([x_zeros, x_ones])
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)])
    x, y = shuffle_np(x, y)

    return x, y

def load_pickle_file(data_path: str, file_name: str = "dump.pickle"):
    with open(os.path.join(data_path, file_name), 'rb') as file:
        df = pickle.load(file)
    return df


def write_pickle_file(df, data_path: str, file_name: str = "dump.pickle") -> None:
    os.makedirs(data_path, exist_ok=True)
    with open(os.path.join(data_path, file_name), 'wb') as file:
        pickle.dump(df, file)
    log.info(f"Saved data to {data_path}/{file_name}")



def get_val_test_ids():
    return {
        'test_ids': (
            (3, 32),
            (3, 18),
            (1, 27),
            (3, 19),
            (3, 17),
            (2, 21),
            (1, 20),
            (1, 11)
        ),
        'val_ids': (
            (3, 3),
            (2, 10),
            (1, 24),
            (3, 24),
            (1, 32),
            (2, 1),
            (1, 10),
            (1, 16)
        )
    }


def plot_single_CV(x, y):
    fig, ax1 = plt.subplots()
    ax1.plot(x[:, 0])
    ax_2 = ax1.twinx()
    ax_2.plot(x[:, 1], color="red")
    title = "good" if y == 1 else "bad"
    plt.title(title)
    fig.tight_layout()
    plt.show()

class MyScaler:

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, x):
        s_0, s_1, s_2 = x.shape
        self.scaler.fit(x.reshape(-1, s_2))

    def transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.transform(x.reshape(-1, s_2))
        return x.reshape(s_0, s_1, s_2)

    def inverse_transform(self, x):
        s_0, s_1, s_2 = x.shape
        x = self.scaler.inverse_transform(x.reshape(-1, s_2))
        return x.reshape(s_0, s_1, s_2)
    
def select_random_val_test_ids():
    mixed = [2,16]
    good_exmples = [2,3,22,24,26,27,28]
    bad_examples = [16,5,7,8,9,10,11,13,14,15,20,21,23,30,31,32]

    good_val_id, good_test_id = np.random.choice(good_exmples, 2, replace=False)
    bad_val_id, bad_test_id = np.random.choice(bad_examples, 2, replace=False)
    return good_val_id, bad_val_id, good_test_id, bad_test_id

def get_data_path():
    return "data"
