import pickle as pkl
import matplotlib.pyplot as plt

import sys
import numpy as np

sys.path.append('../python')

from sepws.nn.lstm import ModelTrainer, CNNLSTM, CNN

from sepws.timeseries.util import time_series_mean, time_series_std, time_series_normalise, time_series_select_feat_variance_thresh, time_series_select_features

from sepws.dataprocessing.ucf import read_train_test

X_train, y_train, X_test, y_test = read_train_test()

for i, x in enumerate(X_train):
    X_train[i] = np.swapaxes(x, 0, 2).swapaxes(1, 2)[::2, :, :]
for i, x in enumerate(X_test):
    X_test[i] = np.swapaxes(x, 0, 2).swapaxes(1, 2)[::2, :, :]



mu = time_series_mean(X_train)
std = time_series_std(X_train)

std[std == 0.0] = 1.0


X_train = time_series_normalise(X_train, mu, std)
X_test = time_series_normalise(X_test, mu, std)


im_size = X_train[0].shape[1:]

conv_ker = [5, 3, 3, 3]
pool_ker = [2, 2, 1, 1]
conv_stride = [2, 1, 1, 1]
pool_stride = [2, 2, 1, 1]
lin_sizes = [2048, 2048, 512]
conv_out_ch = [48, 128, 256, 256]

cnn = CNN(im_size=im_size, in_channels=1, kernel_sizes=conv_ker, out_channels=conv_out_ch, 
          conv_strides=conv_stride, pool_sizes=pool_ker, pool_strides= pool_stride, fully_connected_sizes=lin_sizes)

lstm = CNNLSTM(cnn, lstm_size=512, num_classes=11)

trainer = ModelTrainer(lstm)

trainer.train_model(X_train, y_train, X_test, y_test, 100, batch_size=128, lr=1e-3)