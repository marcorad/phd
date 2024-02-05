
import sys


sys.path.append('../python')

import phd.scattering.config as config
# config.MORLET_DEFINITION = config.MORLET_DEF_DEFAULT
# config.MORLET_DEFINITION = config.MorletDefinition(3, 2, 2, 3, 4)
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

from kymatio.torch import Scattering2D

import skimage as ski



from sklearn import datasets, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from mnist import MNIST
mndata = MNIST('../../python-mnist/data')
images, labels = mndata.load_training()

images = np.array(images)[0::1, :].reshape((-1, 28, 28)).astype(config.NUMPY_REAL)
labels = np.array(labels)[0::1]

images = (images - np.mean(images))/np.std(images)

print(images.shape)


# Create a classifier: a support vector classifier
clf = svm.SVC(cache_size=512, verbose=True)
# clf = LogisticRegression()
# clf = LinearDiscriminantAnalysis()

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, shuffle=False
)

print(X_train.shape)
print(X_test.shape)

t0 = time()

ws = Scattering2D(3, (28, 28))

S_train = ws.scattering(torch.from_numpy(X_train))
S_test  = ws.scattering(torch.from_numpy(X_test))



print(S_train.shape)

torch.cuda.synchronize()

t1 = time()

print("Scattering took {:.2f} ms".format((t1 - t0)*1000))

# S_train = ws.dct(S_train)
# S_test = ws.dct(S_test)

print(S_train.shape)

S_train = S_train.cpu().flatten(start_dim=1).numpy()
S_test = S_test.cpu().flatten(start_dim=1).numpy()


mu = np.mean(S_train, axis=0)
std = np.std(S_train, axis=0)
S_train = (S_train - mu)/std
S_test = (S_test - mu)/std

# u, s, v = np.linalg.svd(S_train.T, full_matrices=False)
# S_train = v.T
# S_test = np.matmul(S_test, u/s[None, :])

print(S_train.shape)
print(S_test.shape)

# # Learn the digits on the train subset
# clf.fit(X_train.reshape((-1, 64)), y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test.reshape((-1, 64)))

# Learn the digits on the train subset
clf.fit(S_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(S_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted, digits=3)}\n"
)




