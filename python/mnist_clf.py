
import sys


sys.path.append('../python')

import sepws.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT


from sepws.scattering.separable_scattering import SeparableScattering
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

from sepws.scattering.config import cfg

cfg.cuda()
cfg.set_beta(1, 2.5)
cfg.set_alpha(1, 2.5)


from sklearn.preprocessing import normalize

torch.cuda.empty_cache()
from kymatio.torch import Scattering2D


TRAIN_SIZES = [300,1000,2000,5000,10000,20000,40000,60000]
Q_CONFIGS = [
    [[1, 1]],
    [[1, 1], [1, 1]]
]

results = {}

for iq, Q in enumerate(Q_CONFIGS):
    d = [8]*2
    print(d)

    ws = SeparableScattering([28, 28], d, Q)


    from sklearn import datasets, metrics, svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import train_test_split

    from mnist import MNIST
    mndata = MNIST('../../python-mnist/data')
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()
    X_train = torch.from_numpy(np.array(X_train).reshape((-1, 28, 28))).type(cfg.REAL_DTYPE)
    y_train = np.array(y_train)
    X_test = torch.from_numpy(np.array(X_test).reshape((-1, 28, 28))).type(cfg.REAL_DTYPE)
    y_test = np.array(y_test)

    # normalise the images
    # mu = np.mean(X_train)
    # sigma = np.std(X_train)
    # X_train = (X_train)/sigma
    # X_test = (X_test)/sigma

    # SVM classifier with multiclass support
    
    torch.cuda.empty_cache()

    #extract features with SWS
    norm = False
    t0 = time()
    S_train_sep = ws.scattering(X_train.to(cfg.DEVICE), normalise=norm).cpu()
    S_test_sep  = ws.scattering(X_test.to(cfg.DEVICE), normalise=norm).cpu()
    torch.cuda.synchronize()
    t1 = time()
    print("Sep Scattering took {:.2f} ms".format((t1 - t0)*1000))
    print(S_train_sep.shape)
    
    # ws_2d = Scattering2D(J=3, shape=(28, 28), max_order=iq+1)
    # ws_2d.cuda()
    
    # t0 = time()
    # S_train_2d: torch.Tensor = ws_2d.scattering(X_train)
    # S_test_2d: torch.Tensor  = ws_2d.scattering(X_test)   
    # torch.cuda.synchronize()
    # t1 = time()
    # print("2D Scattering took {:.2f} ms".format((t1 - t0)*1000))
    # S_train_2d = S_train_2d.swapaxes(1, -1)
    # S_test_2d = S_test_2d.swapaxes(1, -1)
    # print(S_train_2d.shape)
    # print('2D DEVICE', S_test_2d.device)

    # #to numpy
    S_train_sep: np.ndarray = S_train_sep.cpu().numpy()
    S_test_sep: np.ndarray = S_test_sep.cpu().numpy()
    # S_train_2d: np.ndarray = S_train_2d.cpu().numpy()
    # S_test_2d: np.ndarray = S_test_2d.cpu().numpy()

    # #perform Mallat's L2 norm
    # # norm = np.max(np.sqrt(np.sum(S_train**2, axis=(1, 2), keepdims=True)), axis=3, keepdims=True)
    # # S_train /= norm
    # # norm = np.max(np.sqrt(np.sum(S_test**2, axis=(1, 2), keepdims=True)), axis=3, keepdims=True)
    # # S_test /= norm

    # #flatten
    S_train_sep = S_train_sep.reshape(S_train_sep.shape[0], np.prod(S_train_sep.shape[1:]))
    S_test_sep = S_test_sep.reshape(S_test_sep.shape[0], np.prod(S_test_sep.shape[1:]))
    # S_train_2d = S_train_2d.reshape(S_train_2d.shape[0], np.prod(S_train_2d.shape[1:]))
    # S_test_2d = S_test_2d.reshape(S_test_2d.shape[0], np.prod(S_test_2d.shape[1:]))
    
    # print(S_train_sep.shape)
    # print(S_train_2d.shape)
    
    
    
    # np.savetxt('MNISTsample.csv', (S_train[1:2, :].T == S_train[1:2, :]*(1 - np.eye(S_train.shape[1]))), fmt='%d')

    for train_size in TRAIN_SIZES: 
        
        #select the training examples
        S_train_sel = S_train_sep[:train_size, :]
        y_train_sel = y_train[:train_size]  

        #normalise the features
        mu = np.mean(S_train_sel, axis=0)
        std = np.std(S_train_sel, axis=0)
        S_train_sel = (S_train_sel-mu)/std
        S_test_n = (S_test_sep-mu)/std

        #train the model
        # clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.001, priors=[1/10]*10)
        clf = svm.SVC()
        clf.fit(S_train_sel, y_train_sel)

        #predict
        y_pred = clf.predict(S_test_n)
        acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
        
        results[(iq, train_size, 'sep')] = acc

        print(f'Sep {acc=}')
        
        #  #select the training examples
        # S_train_sel = S_train_2d[:train_size, :]
        # y_train_sel = y_train[:train_size]  

        # #normalise the features
        # mu = np.mean(S_train_sel, axis=0)
        # std = np.std(S_train_sel, axis=0)
        # S_train_sel = (S_train_sel-mu)/std
        # S_test_n = (S_test_2d-mu)/std

        # #train the model
        # clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        # clf.fit(S_train_sel, y_train_sel)

        # #predict
        # y_pred = clf.predict(S_test_n)
        # acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
        
        # results[(iq, train_size, '2d')] = acc

        # print(f'2D {acc=}')
        
import pprint

pprint.pprint(results)