"""
Utilities for data modification. 

Arguments named "X" often take features (ie. x_train, x_test) as inputs;
Arguments named "y" often take model or sampling outputs (ie. y_true, y_pred) as inputs.
Name "data" means either features or outputs would be possible input to the function.
"""

import numpy as np
import numba

import warnings
from itertools import combinations

from .operation_utils import sign


@numba.jit(nopython=True)
def add_cons(X):
    """Add constant terms into features

    Parameters
    ----------
    X: 2-D array (n_samples, n_features)
            The data to fit.
    
    Returns
    -------
    output: 2-D array (n_samples, n_features + 1)
            Array of features with the first column added as all '1'. 
    """
    if len(X.shape) != 2:
        raise ValueError("Invalid data: not for 1d-array with or array more than 2 dimensions.")

    n = X.shape[0]
    cons_feature = np.ones(n).reshape((n, 1))
    return np.hstack((cons_feature, X))



def to_column(y):
    """Coerce input into a column vector(2-D, shape = n*1)

    Parameters
    ----------
    y: list, tuple, or 1-D array
    
    
    Returns
    -------
    output: 2-D array (n_sample, 1)
    """
    return np.asarray(y).reshape((-1, 1))



def to_vector(y):
    """Convert data into an 1-D array.
    
    Parameters
    ----------
    y: list, tuple, or n-D array
    

    Returns
    -------
    output: 1-D array (n_sample, )
    """
    return np.asarray(y).flatten()



def one_hot_encode(data):
    """Convert categorical data to numerical data using one-hot-encoding.

    Parameters
    ----------
    data: list, tuple, or 1-D array
    

    Returns
    -------
    output: 2-D array 
            One-hot-encoded data. The class stored in corresponding
            label is assigned the value 1 and the others 0.
    
    labels: 1-D array
            The tag of each class encoded as 1 in the
            corresponding column.
    """
    labels = np.unique(data)
    output = np.empty((len(data), len(labels)))
    for i in range(len(labels)):
        is_class = sign(data == labels[i])
        output[:, i] = is_class
    
    return output, labels



def create_ova_data(y):
    """Create data for one-versus-all multiclass classification.

    Parameters
    ----------
    y: 1-D array


    Returns
    --------
    output: 2-D array
            Which is actully one-hot-encoded data.
    """
    return one_hot_encode(y)



def create_ovo_data(X, y):
    """Convert data into one-versus-one form.

    Parameters
    ----------
    X: 2-D array.
            Features in training data.
    
    y: 1-D array.
            Sample values in training data.

    Returns
    -------
    output: dictionary.
            Key: tuples which their first element stores
                the class name encoded as +1 and the second -1.
            Value: tuples. The first element stores feature
                array. The second element stores value array.
    """
    labels = list(np.unique(y))
    label_combs = list(combinations(labels, 2))
    # i[0] stores the class name encoded as +1
    # i[1] stores the class name encoded as -1

    xs = [X[np.isin(y, i), :] for i in label_combs]
    ys = [sign(to_column(y)[np.isin(y, i), :] == i[0]) for i in label_combs]
    output = {i:(x, y) for i, x, y in zip(label_combs, xs, ys)}
    return output



def re_weight_data(data, weight, method=["sampling", "copying"]):
    """Weightedly sample the training data.


    Parameters
    ----------
    data:   An array or a tuple with 2 arrays.
            If the input is a tuple, the first element should be an
            array of features and the second should be an array of 
            labels or values.
    
    weight: 1-D array like.
            The weight used to resample each corresponding rows.

    method: "sampling" or "copying"
            If the method is "sampling", each rows will be drew
            based on the weight parameter and remain the number
            of samples unchanged. 
            If the method is "copying", each rows will repeat
            by the time given in weight. This might return a 
            greater training data set.

    Returns
    -------
    output: An array or a tuple with 2 arrays.
            Based on the given data.
    """
    if type(data) is tuple:
        X, y = data
        data = np.hstack((X, y[:,None]))
        input_is_tuple = True 
    else:
        input_is_tuple = False

    if method == "sampling":
        random_id = np.random.choice(
            a = np.arange(data.shape[0]),
            size = data.shape[0],
            p = np.array(weight)/np.sum(weight)
        )
        output = data[random_id, :]

    if method == "copying":
        output = np.repeat(data, weight, axis=0)
    
    if input_is_tuple:
        return output[:, 0:-1], output[:, -1]
    else:
        return output




def create_transformation(X, feature_idx=None, weight=None):
    """Create transformation matrix.
    
    A transformation matrix created during the random forest
    procedure is a sparse matrix (a matrix full of zeros) with
    non-zero elements assigned the value of weight coeffients.
    By multiplying the transformation matrix, you can sample the
    feature space and sum up the sampled features with given weight
    at the same time.

    Parameters
    ----------
    X: 2-D array (n_samples, n_features)
        Features

    feature_idx: None or 1-D array
        The indices of features to be selected from original
        feature space. If `feature_idx==None`, create no
        transformation (namely, identical transformation).
    
    weight: None or array (n_feature_idx, n_created_feature)
        The weight to apply when combine the sampled features.
        If `weight==None`, this function will create a matrix
        that simply maps the sampled feature without any linear
        combination.

    Returns
    -------
    transformation: 2-D array
            The transformation matrix.
    """
    if feature_idx is None:
        # No transformation
        transformation = np.identity(X.shape[1])
    else:
        feature_idx = np.array(feature_idx)
        
        if weight is None:
            # feature sampling without linear combination
            n_feat = len(feature_idx)
            transformation = np.zeros([X.shape[1], n_feat])
            transformation[feature_idx, np.arange(n_feat)] = 1
        else:
            # feature sampling and combination
            n_feat = weight.shape[1]
            transformation = np.zeros([X.shape[1], n_feat])
            transformation[feature_idx.reshape(-1,1), np.arange(n_feat)] = weight 
    
    return transformation



def permutate(X, feature_idx):
    """Change the order of values in the given feature.
    Other columns remain the same.

    Parameters
    ----------
    X: 2-D array
        The data to modify, possibly the out-of-bag data
        from random forest model to evaluate feature 
        importance.
    
    feature_idx: int
        The feature, or column, to shuffle order
    
    Returns
    -------
    output: 2-D array
        The data which a random permutation has
        done to `col==feature_idx` 
    """

    output = X
    x_feature = X[:, feature_idx]
    np.random.shuffle(x_feature)
    output[:, feature_idx] = x_feature
    
    return output

