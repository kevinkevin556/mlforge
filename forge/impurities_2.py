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
    
    @abc.abstractmethod
    def solve_split(self, y):
        pass


class BinaryMisclassification():
    __problem_type__ = "binary_classification"
    
    @staticmethod
    @njit
    def eval(y):
        N = len(y)
        values, counts = unique_count(y)
        most_freq_value = values[np.where(counts==np.max(counts))]
        return 1 - np.sum(y==most_freq_value)/N

    @staticmethod
    def solve_split(y_sorted):
        """ Find out the split which minimizes binary
        misclassification error for 2 classes. 
        The recursive approach is applied and thus the 
        complexity of the algorithm can be limited
        within `O(N)`.

        Parameters
        ----------
        y_sorted: 1-D array like.
            The label permutation based on any 
            sorted feature in training data.

        Returns
        -------
        split:  dictionary, 
            Include the information of the best split, 
            See  _recursive_solve() for exmaples.

            sign: +1 or -1 
            position: int, in [0, n_sample+1]
            error: sum of impurity from 2 parts,
                namely, the misclassification error
        """
        split = None
        positive_ray, negative_ray, _ = \
            BinaryMisclassification.recursively_solve(np.array(y_sorted), len(y_sorted))
            
        if positive_ray["error"] < negative_ray["error"]:
            split = positive_ray
        else:
            split = negative_ray
        return split["position"], split["error"]


    @staticmethod
    @njit
    def recursively_solve(partition, partition_length):
        """ Solve the minimizing impurity problem in divide and conquer
        approach.

        Parameters
        ----------
        partition: 1-D array
            A sequence of labels.

        partition_length: int
            The length of partition.

        Returns
        -------
        split: dictionary
            See examples.
        
        count: dictionary 
            Numbers of positive and negative labels


        Examples for dict of split
        --------------------------
            ...-|+...    (split)
        1. - - -|+ + +   (sample)
            positive ray hypothesis with cut at position 3
                {"sign": 1, "position": 3, "error": 0}   

           ..+|-...      (split)
        2. + +|- - - +   (sample)
            negative ray hypothesis with cut at position 2
                {"sign": -1, "position": 2, "error": 1}
        """

        if partition_length == 1:            
            if partition[0] == 1:
                #        split = [sign, position, error]
                positive_ray = {"sign":  1, "position": 0, "error": 0}
                negative_ray = {"sign": -1, "position": 1, "error": 0}
                counts = {+1: 1, -1: 0}
            
            if partition[0] == -1:
                positive_ray = [ 1, 1, 1] {"sign":  1, "position": 1, "error": 0}
                negative_ray = [-1, 0, 0] {"sign":  -1, "position": 0, "error": 0}
                counts = {+1: 0, -1: 1}
        else: 
            # Divide
            midpoint = partition_length // 2
            front_partition, back_partiton = partition[0:midpoint], partition[midpoint:partition_length]
            
            # Conquer
            front_pos_ray, front_neg_ray, front_counts = \
                BinaryMisclassification.recursively_solve(front_partition, midpoint)
            back_pos_ray, back_neg_ray, back_counts = \
                BinaryMisclassification.recursively_solve(back_partiton, partition_length-midpoint)
            
            positive_ray = \
                BinaryMisclassification.find_best_split(1, front_pos_ray, front_counts, back_pos_ray, back_counts, midpoint)
            negative_ray = \
                BinaryMisclassification.find_best_split(-1, front_neg_ray, front_counts, back_neg_ray, back_counts, midpoint) 

            counts = {
                +1: front_counts[+1] + back_counts[+1],
                -1: front_counts[-1] + back_counts[-1]
            }
        
        return positive_ray, negative_ray, counts


    @staticmethod
    @njit
    def find_best_split(sign, front_split, front_counts, back_split, back_counts, midpoint):
        output = {}
        output["sign"] = sign

        error_applying_front_split = front_split["error"] + back_counts[-sign]
        error_applying_back_split = back_split["error"] + front_counts[sign]

        if error_applying_front_split < error_applying_back_split:
            output["position"] = front_split["position"]
            output["error"] = error_applying_front_split
        else:
            output["position"] = back_split["position"] + midpoint
            output = error_applying_back_split

        return output



class MeanSquaredError(Impurity):
    __problem_type__ = "regression"

    @staticmethod
    @njit
    def eval(y):
        return np.mean((y-np.mean(y))**2)

    @staticmethod
    @njit
    def solve_split(y_sorted):
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
    def eval(y):
        N = y.shape[0]
        _, counts = unique_count(y)
        return 1 - np.sum((counts/N)**2)

    @staticmethod
    @njit
    def solve_split(y_sorted):
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
