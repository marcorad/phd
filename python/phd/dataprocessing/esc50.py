import csv
import scipy.io.wavfile as wav
import scipy.signal as sig
from math import floor
from tqdm import tqdm

from ..scattering import config
import pickle as pkl
import numpy as np

from typing import List, Dict

DATASET_DIR = '/media/data/Datasets/ESC-50-master/'


def read_meta():
    meta: Dict[str, List[str]] = {}
    with open(DATASET_DIR + 'meta/esc50.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['category']
            if label not in meta.keys(): meta[label] = []
            meta[label].append(row['filename'])
    return meta

AUDIO_PATH = DATASET_DIR + 'audio/'

def pre_process(target_fs, train_prop = 0.8):
    meta = read_meta()
    fs = 44100
    decimate = floor(fs/target_fs)
    fs_true = fs / decimate
    print(f'Resampling to {fs_true} Hz')
    print(f'Dataset contains {len(meta.values())} samples.')  
    X_train, X_test = [], []  
    y_train, y_test = [], []  
    N_train_items = floor(train_prop*len(list(meta.values())[0]))
    for label in tqdm(meta.keys()):
        n = 0
        for fname in meta[label]:            
            _, x = wav.read(AUDIO_PATH + fname)
            x = x.astype(config.NUMPY_REAL)       
            x = x/(2**15-1)
            x = sig.resample_poly(x, up=1, down=decimate)
            # xn = (x - np.mean(x))
            # xn /= np.std(x[x!=0])
            if n < N_train_items:
                X_train.append(x)
                y_train.append(label)
            else:
                X_test.append(x)
                y_test.append(label)
            n += 1
        wav.write(AUDIO_PATH + 'processed/' + fname, floor(fs_true), x)
    X_train = np.array(X_train, dtype=config.NUMPY_REAL)
    X_test = np.array(X_test, dtype=config.NUMPY_REAL)
    with open('data/esc50.pkl', 'wb') as file:
        pkl.dump((X_train, y_train, X_test, y_test), file)

def load_esc50():
    with open('data/esc50.pkl', 'rb') as file: 
        X_train, y_train, X_test, y_test = pkl.load(file)
    return X_train, y_train, X_test, y_test


    

    



