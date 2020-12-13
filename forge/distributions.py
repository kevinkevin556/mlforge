import numpy as np
from numpy import pi, exp
from numpy.linalg import det, inv
from numba import float64 
from numba.experimental import jitclass

from .utils.operation_utils import mean


class Distribution():
    def __init__(self):
        pass

    def proba(self, x):
        pass

    def fit(self, X):
        pass
    
    def copy(self):
        pass


@jitclass([("mean", float64[:]), ("covariance", float64[:, :])])
class Gaussian(Distribution):
    def __init__(self, mean=np.empty(0), covariance=np.empty((0,0))):
        self.mean = mean
        self.covariance = covariance


    def proba(self, x):
        d = x.shape[1]
        mu = self.mean
        sigma = self.covariance
        return np.diag(np.power(2*pi, -d/2) * np.power(det(sigma), -1/2) * exp(-1/2 * (x-mu) @ inv(sigma) @ (x-mu).T))


    def fit(self, X):
        self.mean = mean(X, axis=0)
        self.covariance = (X-self.mean).T @ (X-self.mean) / X.shape[0]
        return self


    def copy(self):
        return Gaussian(self.mean.copy(), self.covariance.copy())