import pickle as pkl
import matplotlib.pyplot as plt

import sys

sys.path.append('../python')

from phd.nn.lstm import BasicLSTM, ModelTrainer, CNNLSTM

from phd.timeseries.util import time_series_mean, time_series_std, time_series_normalise, time_series_select_feat_variance_thresh, time_series_select_features

feat_file_name = 'data/ws-ucf-Q=[[2, 2, 2]]-T=[48.0, 48.0, 0.8]-kernel_size=1.pkl'

with open(feat_file_name, 'rb') as file:
    X_train, y_train, X_test, y_test = pkl.load(file)
    
# idx = time_series_select_feat_variance_thresh(X_train, 8192)
# X_train = time_series_select_features(X_train, idx)
# X_test = time_series_select_features(X_test, idx)
    
mu, std = time_series_mean(X_train), time_series_std(X_train)
X_train = time_series_normalise(X_train, mu, std)
X_test = time_series_normalise(X_test, mu, std)
    
print(X_train[0].shape[1])
model = CNNLSTM(X_train[0].shape[1], 256, 11, dropout=0.2)
mt = ModelTrainer(model, reg_norm=2)

import traceback

try:
    mt.train_model(X_train, y_train, X_test, y_test, batch_size=128, lr=1e-4, n_epochs=100, lambda_=0e-6)
except: # Exception as e:
    # traceback.print_exception(e)
    plt.plot(mt.training_history)
    plt.legend(['Training', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show(block='True')
    
