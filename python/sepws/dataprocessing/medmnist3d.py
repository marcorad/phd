import numpy as np
from typing import Tuple

DATASET_DIR = '/media/data/Datasets/'
DATASETS = ['organ', 'adrenal', 'fracture', 'nodule', 'vessel', 'synapse']


def load_train_test(dataset, combine_val_in_train = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    is_valid = dataset in DATASETS
    assert is_valid, f'{dataset} not in MedMNIST3D datasets. Valid options are: {"; ".join(DATASETS)}.'
    
    folder_path = DATASET_DIR + dataset + '-mnist-3d/'
    X_train, y_train = np.load(folder_path + 'train_images.npy'), np.load(folder_path + 'train_labels.npy')[:, 0]
    X_test, y_test = np.load(folder_path + 'test_images.npy'), np.load(folder_path + 'test_labels.npy')[:, 0]
    X_val, y_val = np.load(folder_path + 'val_images.npy'), np.load(folder_path + 'val_labels.npy')[:, 0]
    if combine_val_in_train:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        return X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test, X_val, y_val


if __name__ == '__main__':
    # load_train_test('asfasfasf')
    for d in DATASETS:
        X_train, y_train, X_test, y_test = load_train_test(d, True)
        print(d, X_train.shape, y_train.shape, X_test.shape, y_test.shape)