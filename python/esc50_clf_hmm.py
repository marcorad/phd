

import pickle as pkl

from hmmlearn import hmm
import numpy as np
from typing import Dict
from tqdm import tqdm



filename = 'data/ws-esc50-[018575963718820862, 320]-20-4-[[2, 1]].pkl'
with open(filename, 'rb') as file:
    X_train, y_train, X_test, y_test = pkl.load(file)
    
labels = set(y_train)
label_sel = list(labels)
labels = label_sel
X_train = X_train[[y in labels for y in y_train], :, :]
X_test = X_test[[y in labels for y in y_test], :, :]
y_train = [y for y in y_train if y in labels]
y_test = [y for y in y_test if y in labels]

#normalise each time slice
# X_train = X_train / np.var(X_train, axis=-1, keepdims=True)
# X_test = X_test / np.var(X_test, axis=-1, keepdims=True)

print(np.sum(np.isnan(X_train)))

hmms: Dict[str, hmm.GaussianHMM] = {}
feat_idx = np.argsort(np.std(X_train, axis=(0, 1)))[:]
mu = np.median(X_train[:, :, feat_idx], axis=(0, 1), keepdims=True)
std = np.median(np.abs(X_train[:, :, feat_idx] - mu), axis=(0, 1), keepdims=True)


n_states = 20

counter = 0

for l in tqdm(labels):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', verbose=False, min_covar=1e-3, tol=1e-1, n_iter=100, algorithm='viterbi')
    #initialise the model
    # x = X_train[np.array([y == l for y in y_train]), ...]    
    
    # # x = x / norm
    # x = x[:, :, feat_idx]    
    # x = (x - mu)/std
    
    # vars = np.var(x, axis=(0,1))/n_states
    # vars = np.stack([vars for _ in range(n_states)], axis=0)
    # mus = np.percentile(x, q=[5 + i*90/(n_states-1) for i in range(n_states)], axis=(0,1))
    # model.covars_ = vars
    # model.means_ = mus
    # model.transmat_ = np.array([[1/n_states for _ in range(n_states)] for _ in range(n_states)])
    # model.startprob_ = np.array([1/n_states for _ in range(n_states)])
    hmms[l] = model
    
#fit each HMM model
counter = 0
for l in labels:
    print(l)
    model = hmms[l]
    x = X_train[np.array([y == l for y in y_train]), ...]
    x = x[:, :, feat_idx]
    x = (x - mu)/std
    L = x.shape[1]
    N = x.shape[0]
    xt = np.concatenate([x[i, ...] for i in range(x.shape[0])], axis=0)
    hmms[l] =model.fit(xt, lengths=[L]*N)
    state_counter = n_states - 1
    while not model.monitor_.converged or np.any(np.sum(model.transmat_, axis=1) == 0) or model.monitor_.history[-1] - model.monitor_.history[-2] < -1:
        print(f'Model for {l} did not converge. Trying {state_counter} states.')
        model = hmm.GaussianHMM(n_components=state_counter, covariance_type='diag', verbose=False, min_covar=1e-3, tol=1e-1, n_iter=100, algorithm='viterbi')
        # vars = np.var(x, axis=(0,1))/state_counter
        # vars = np.stack([vars for _ in range(state_counter)], axis=0)
        # mus = np.percentile(x, q=[5 + i*90/(state_counter-1) for i in range(state_counter)], axis=(0,1))
        # model.covars_ = vars
        # model.means_ = mus
        # model.transmat_ = np.array([[1/state_counter for _ in range(state_counter)] for _ in range(state_counter)])
        # model.startprob_ = np.array([1/state_counter for _ in range(state_counter)])
        hmms[l] = model.fit(xt, lengths=[L]*N)
        state_counter -= 1
        
    
#predict the class from the model with the highest score
preds = []
n_correct = 0
N = X_test.shape[0]
counter = 0
for i, y in enumerate(y_test):
    yscore = []
    x = X_test[i, ...]
    x = x[:, feat_idx]    
    x = (x - mu[0,...])/std[0, ...]
    for l in labels:
        # x = x / norm[0, ...]
        yscore.append(hmms[l].score(x))
    yp = list(labels)[np.argmax(yscore)]
    preds.append(yp)
    print(f'True: {y}, Predicted: {yp}')
    if y == yp:
        n_correct += 1
        
print(f'Accuracy: {n_correct / N}')
        
        
    
