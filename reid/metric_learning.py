from __future__ import absolute_import

import torch

from .evaluators import extract_features

from metric_learn.base_metric import BaseMetricLearner

from metric_learn import (ITML_Supervised, LMNN, LSML_Supervised,
                          SDML_Supervised, NCA, LFDA, RCA_Supervised)



class Euclidean(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self, X):
        self.M_ = np.eye(X.shape[1])
        self.X_ = X

    def transform(self, X=None):
        if X is None:
            return self.X_
        return X

def validate_cov_matrix(M):
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    while True:
        try:
            _ = np.linalg.cholesky(M)
            break
        except np.linalg.LinAlgError:
            # Find the nearest positive definite matrix for M. Modified from
            # http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            # Might take several minutes
            k += 1
            w, v = np.linalg.eig(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M


class KISSME(BaseMetricLearner):
    def __init__(self):
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self, X, y=None):
        n = X.shape[0]
        if y is None:
            y = np.arange(n)
        X1, X2 = np.meshgrid(np.arange(n), np.arange(n))
        X1, X2 = X1[X1 < X2], X2[X1 < X2]
        matches = (y[X1] == y[X2])
        num_matches = matches.sum()
        num_non_matches = len(matches) - num_matches
        idxa = X1[matches]
        idxb = X2[matches]
        S = X[idxa] - X[idxb]
        C1 = S.transpose().dot(S) / num_matches
        p = np.random.choice(num_non_matches, num_matches, replace=False)
        idxa = X1[~matches]
        idxb = X2[~matches]
        idxa = idxa[p]
        idxb = idxb[p]
        S = X[idxa] - X[idxb]
        C0 = S.transpose().dot(S) / num_matches
        self.M_ = np.linalg.inv(C1) - np.linalg.inv(C0)
        self.M_ = validate_cov_matrix(self.M_)
        self.X_ = X






class DistanceMetric(object):
    def __init__(self, algorithm='euclidean', *args, **kwargs):
        super(DistanceMetric, self).__init__()
        self.algorithm = algorithm
        self.metric = get_metric(algorithm, *args, **kwargs)

    def train(self, model, data_loader):
        if self.algorithm == 'euclidean': return
        features, labels = extract_features(model, data_loader)
        features = torch.stack(features.values()).numpy()
        labels = torch.Tensor(list(labels.values())).numpy()
        self.metric.fit(features, labels)

    def transform(self, X):
        if torch.is_tensor(X):
            X = X.numpy()
            X = self.metric.transform(X)
            X = torch.from_numpy(X)
        else:
            X = self.metric.transform(X)
        return X

__factory = {
    'euclidean': Euclidean,
    'kissme': KISSME,
    'itml': ITML_Supervised,
    'lmnn': LMNN,
    'lsml': LSML_Supervised,
    'sdml': SDML_Supervised,
    'nca': NCA,
    'lfda': LFDA,
    'rca': RCA_Supervised,
}

def get_metric(algorithm, *args, **kwargs):
    if algorithm not in __factory:
        raise KeyError("Unknown metric:", algorithm)
    return __factory[algorithm](*args, **kwargs)