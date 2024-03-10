import numpy as np
from typing import Tuple

DATASET_DIR = '/media/data/Datasets/organ-mnist-3d/'

def load_train_test(combine_val_in_train = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = np.load(DATASET_DIR + 'train_images.npy'), np.load(DATASET_DIR + 'train_labels.npy')[:, 0]
    X_test, y_test = np.load(DATASET_DIR + 'test_images.npy'), np.load(DATASET_DIR + 'test_labels.npy')[:, 0]
    X_val, y_val = np.load(DATASET_DIR + 'val_images.npy'), np.load(DATASET_DIR + 'val_labels.npy')[:, 0]
    if combine_val_in_train:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        return X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_train_test(True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    X_train, y_train, X_test, y_test, X_val, y_val = load_train_test(False)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)
    