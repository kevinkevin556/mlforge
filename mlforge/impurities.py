import abc
import numpy as np
from numba import njit
from numba.experimental import jitclass

from .utils.operation_utils import unique_count
from .metrics import Metric


class Impurity(Metric, metaclass=abc.ABCMeta):
    __problem_type__ = None

    @abc.abstractstaticmethod
    def eval(self, y):
        pass



class MeanSquaredError(Impurity):
    __problem_type__ = "regression"

    @staticmethod
    @njit
    def eval(y):
        return np.mean((y-np.mean(y))**2)

    @staticmethod
    @njit
    def find_split(y_sorted):
        """Find out the split which minimizes MSE.
        The presented trick using variance gain can
        reach the complexity of `O(N)` for this problem.

        
        Parameters
        ----------
        y_sorted: 1-D array like.
            The label permutation based on any 
            sorted feature in training data.

        Returns
        -------
        split:  dictionary, 
            Include the information of the best split. 

            position: int, in [0, n_sample+1]
            error: sum of impurity from 2 parts, 
                which equals weighted sum of MSE.
        """
        
        position = 0                 # initialize with the situation before split(no split, pos=0) 
        sum_left = 0                 # Sum of left partition
        sum_right = np.sum(y_sorted) # ...... right ........
        n_left = 0                   # Number of elements in the left partition
        n_right = len(y_sorted)      # ......................... right ........
        
        max_variance_gain = -np.inf
        
        for i in range(len(y_sorted)-1): # len()-1 to assure the cut is somewhere in between 
            n_left  += 1
            n_right -= 1

            sum_left  += y_sorted[i]
            sum_right -= y_sorted[i]
            
            variance_gain = (sum_left**2)/n_left + (sum_right**2)/n_right
            
            if variance_gain > max_variance_gain:
                max_variance_gain = variance_gain
                position = i + 1
                
        min_mse = np.sum(y_sorted**2) - max_variance_gain
        min_impurity_sum = min_mse

        return position, min_impurity_sum



class GiniIndex(Impurity):
    __problem_type__ = "binary_classification"

    @staticmethod
    @njit
    def eval(y):
        N = y.shape[0]
        _, counts = unique_count(y)
        return 1 - np.sum((counts/N)**2)


    @staticmethod
    @njit
    def find_split(y_sorted):
        """Find out the split which minimizes weighted 
        sum of gini index for 2 classes.

        See: https://www.csie.ntu.edu.tw/~htlin/course/mltech20spring/doc/209_handout.pdf (P.12) 

        Parameters
        ----------
        y_sorted: 1-D array like.
            The label permutation based on any 
            sorted feature in training data.

        Returns
        -------
        split:  dictionary, 
            Include the information of the best split. 

            position: int, in [0, n_sample+1]
            error: sum of impurity from 2 parts,
                which equals weighted sum of Gini index.
        """
                
        position = 0    # initialize with the situation before split(no split, pos=0) 
        counts_left = 0                     # counts of class #1 in the left partition
        counts_right = np.sum(y_sorted==1)  # ......................... right.........
        n_left = 0                          # number of elements in the left partition
        n_right = len(y_sorted)             # ......................... right.........

        gini = lambda mu: mu * (1 - mu)     # formula of gini index in 2-class case
                                            # mu = counts/n in each partition
        min_impurity_sum = np.Inf
        
        for i in range(len(y_sorted)-1): # len()-1 to assure somewhere in between is cut 
            n_left  = n_left  + 1
            n_right = n_right - 1
            
            if y_sorted[i] == 1:
                counts_left  = counts_left  + 1
                counts_right = counts_right - 1

            mu_left  = counts_left  / n_left
            mu_right = counts_right / n_right

            weighted_sum_of_gini = 2 * (n_left * gini(mu_left) + n_right * gini(mu_right))
            
            if weighted_sum_of_gini < min_impurity_sum:
                min_impurity_sum = weighted_sum_of_gini
                position = i + 1
        
        return position, min_impurity_sum


class Entropy():
    __problem_type__ = "binary_classification"

    @staticmethod
    @njit
    def eval(y):
        _, counts = unique_count(y)
        p = counts / y.shape[0]
        return np.sum(-p * np.log2(p))


