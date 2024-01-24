
import sys

sys.path.append('../python')

import phd.scattering.config as config
config.MORLET_DEFINITION = config.MorletDefinition(2, 2, 3, 3, 3)
# config.MORLET_DEFINITION = config.MORLET_DEF_PERFORMANCE
config.set_precision('single')
config.ENABLE_DS = True

from phd.scattering.sep_ws import SeperableWaveletScattering, optimise_T
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

torch.cuda.empty_cache()

fs = [1, 1]
Q = [[1, 1]]
T = [optimise_T(12, 1, eps=0.05)]*2
print(T)

ws = SeperableWaveletScattering(Q, T, fs, [1, 2], True)


# plt.subplot(Np, Np, 1)
# plt.imshow(x.cpu())

# for i in range(min(U.shape[-1], Np*Np-1)):
#     plt.subplot(Np, Np, i + 1)
#     plt.imshow(U.cpu()[:, :, i].real)
#     lambdas = wsl.filter_lambda_pairs[i]
#     plt.title("{:.2f}, {:.2f}".format(*lambdas))

# plt.show()

import skimage as ski



from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, 8, 8))

# Create a classifier: a support vector classifier
clf = svm.SVC()

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

print(X_train.shape)
print(X_test.shape)

S_train= ws.scatteringTransform(torch.from_numpy(X_train))
S_test = ws.scatteringTransform(torch.from_numpy(X_test))

S_train = S_train.cpu().flatten(start_dim=1).numpy()
S_test = S_test.cpu().flatten(start_dim=1).numpy()

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
    f"{metrics.classification_report(y_test, predicted)}\n"
)




