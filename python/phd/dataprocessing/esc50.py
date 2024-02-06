import csv
import scipy.io.wavfile as wav
import scipy.signal as sig
from math import floor
from tqdm import tqdm

from ..scattering import config
import pickle as pkl
import numpy as np

DATASET_DIR = '/media/data/Datasets/ESC-50-master/'



def read_meta():
    meta = {}
    with open(DATASET_DIR + 'meta/esc50.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            meta[row['filename']] = {
                'label': row['category'],
                'esc10': row['esc10']
            }
    return meta


AUDIO_PATH = DATASET_DIR + 'audio/'

def pre_process(target_fs):
    meta = read_meta()
    audio = []
    labels = []
    fs, x = wav.read(AUDIO_PATH + list(meta.keys())[0])
    N = x.shape[0]
    decimate = floor(fs/target_fs)
    fs_true = fs / decimate
    print(f'Resampling to {fs_true} Hz')
    print(f'Dataset contains {len(meta.keys())} samples.')    
    for fname, info in tqdm(meta.items()):
        _, x = wav.read(AUDIO_PATH + fname)
        x = x.astype(config.NUMPY_REAL)       
        x = x/(2**15-1)
        x = sig.resample_poly(x, up=1, down=decimate)
        xn = (x - np.mean(x))
        xn /= np.max(np.abs(xn))
        audio += [xn]
        labels += [info['label']]
        wav.write(AUDIO_PATH + 'processed/' + fname, floor(fs_true), x)
    audio = np.array(audio, dtype=config.NUMPY_REAL)
    with open('data/esc50.pkl', 'wb') as file:
        pkl.dump((audio, labels), file)

def load_esc50():
    with open('data/esc50.pkl', 'rb') as file: 
        audio, labels = pkl.load(file)
    return audio, labels

def get_labels():
    meta = read_meta()
    labels = []
    for v in meta.values():
        labels.append(v['label'])
    return labels
    

    



