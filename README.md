# mlforge

**FORGE** a module for Machine Learning algorithms in Python from scratch.

[![Build Status](https://travis-ci.com/kevinkevin556/pyforge.svg?branch=master)](https://travis-ci.com/kevinkevin556/pyforge)
[![codecov](https://codecov.io/gh/kevinkevin556/pyforge/branch/master/graph/badge.svg)](https://codecov.io/gh/kevinkevin556/pyforge)


#### Modularized for Simplicity and Readablity

> A simplified focus on each of the parts helps us better understand the whole.

The module separates the ML models and their optimizer/solver implementation respectively to offer a clear picture of logic in machine learning algorithm, which was initially inspired by Keras. Users can tell the parameters that determine the performance of model from those which are used to acheieve convergence in training. The design also enables each parts in the module to be independently tested and experimented with many possible combinations.

You can find the implementation of gradient for losses and regularizers in their class, and so does the minimization of impurities.

#### Keras-like API

* `compile()` configures the model for training.
* `fit()` trains the model.
* `evaluate()` returns the value of loss/metric for the model.
* `predict()` generates output predictions for the input samples.

#### Scikit-learn Compatible

For users who are familar with sklearn or demand to practice sklearn api,

* `get_params()`
* `set_param()`
* `score()`

are also provided and operate similarly as their counterpart in sklearn.

~~Besides, users can cross-validate the implemented models with `sklearn.model_selection` or build a pipeline for the algorithms using `sklearn.pipeline`.~~

#### Acceralated with Numba

Although performance is not the primary concern of the module, numba compiler helps this rough implementation in Python run faster.

## Get Started

### Read the source code / Find the implementation of ML algorithm

#### directory structure

In directories of each categories, there are two main py-files: **`models.py`** and **`optimizers.py`**. `models.py` contains designs of machine learning models and `optimizers.py` contains the solvers which work out the optimization problem given by the model.

#### decorators

Decorators are used to show to the essential parts of models and optimizers.

##### in optimizer

* `@implementation`: The core piece of code conducts optimization to the machine learning model. It can work independently if you provide enough parameters and preprocess the training data.

##### in model

* `@fit_method`: the function is called to fit the model.
* `@predict_method`: the function is called to gernerate prediction.

### Create your ML model

All you have to do is to write functions **`_init__()`**, **`fit()`**, **`predict()`** and specify **`model_type`** for your model.

```Python
from mlforge.base.models import Model
from mlforge.utils import fit_method, predict_method

class TestModel(Model):
    model_type = "binary-classifier" # "binary-classifier" or "regressor"

    def __init__(self, optimizer, metrics):
        self.w_ = None                        # initialize the fitted result on your own
        self.compile(
            optimizer = optimizer,            # call self.compile() with keyword arguments
            metrics = metrics                 # to generate attributes for user-specified parameters
        )


    # Adding @fit_method for fit() helps you encode binary inputs into (-1, 1)
    @fit_method
    def fit(self, X, y, **kwargs):
        self.w_ = self.optimzier.excute(X, y, **kwargs)
        return self


    # Adding @predict_method for predict() helps you transform (-1, 1) outputs into original labels
    @predict_method
    def predict(self, X):
        return X @ self.w_
```


## Implemented Models and Algorithms

### [Perceptron]()

Models       | Optimizers                 | Scikit-learn Compatible | Numba-acclerated
---          | ---                        | ---                     | ---
Perceptron   | Linear Separable PLA       | **O**                   | **O**
_            | Pocket Algorithm           | **O**                   | **O**
Adaline      | Gradient Descent (Adaline) | **O**                   | **O**

### [Regression]()

Models                     | Optimizers                      | Scikit-learn Compatible  | Numba-acclerated
---                        | ---                             | ---                      | ---
Linear Regression          | Analytic Solution               | **X**                    | **O**
_                          | Gradient Descent                | **X**                    | **O**
_                          | Stocastic Gradient Descent      | **X**                    | **O**
Ridge Regression           | *`Same as Linear Regression`*   | **X**                    | **O**
Logistic Regression        | Gradient Descent                | **X**                    | **O**
_                          | Stocastic Gradient Descent      | **X**                    | **O**
Kernel Ridge Regression    | *`Same as Linear Regression`*   | **X**                    | **O**
Kernel Logistic Regression | *`Same as Logistic Regression`* | **X**                    | **O**

### [Support Vector Machine]()

Models                    | Optimizers                  |  Scikit-learn Compatible | Numba-acclerated
---                       | ---                         | ---                      | ---
Hard-margin SVM           | Primal QP Solver            | **X**                    | **X**
_                         | Dual QP Solver              | **X**                    | **X**
Soft-margin SVM           | *`Same as Hard-margin SVM`* | _                        | _
Support Vector Regression | *`Same as Hard-margin SVM`* | _                        | _

### [Ensemble Models]()

Models                    | Optimizers
---                       | ---
Bagging Perceptron        | *`Same as Perceptron`*
AdaBoost-Stump            | *`Same as Decision Stump`*

### [Tree Models]()

Models                | Optimizers                 | Scikit-learn Compatible  | Numba-acclerated
---                   | ---                        | ---                      | ---
Decision Stump        | Decision Stump Solver      | **X**                    | **X**
Decision Tree         | ID3                        | **X**                    | **X**
_                     | CART                       | **X**                    | **X**
Random Forest         | RandomTree(tree=CART)      | **X**                    | **X**
Gradient Boosted Tree | _                          | **X**                    | **X**

### [Bayes Models]()

Models                | Optimizers                 | Scikit-learn Compatible  | Numba-acclerated
---                   | ---                        | ---                      | ---
Naive Bayes           | Naive Bayes Solver         | **X**                    | **O**

### [Multiclass Classification Meta-Algorithm]()

* One versus All Decomposition
* One versus One Decomposition

### [Ensemble Meta-algorithms]()

* Voting
* Linear Blending
* Stacking
* Bagging
* AdaBoost


### [Other Implementation]()

#### Loss

* 0-1 Error (Binary Classification Error)
* Mean Squared Error
* Cross-Entropy

#### Kernel

* Linear Kernel
* Polynomial Kernel
* Gaussian Kernel (RBF)

#### Regularization

* L1 regularization
* L2 regularization
* Tikhonov regularization
* (Null: No regularization)

#### Distribution

* Gaussian distribution

## Reference

### Lectures

1. [Learning from data MOOC](https://work.caltech.edu/telecourse.html): Taught by Caltech Prof. Yaser Abu-Mostafa
2. [Machine Learning Foundations--Mathematical Foundations](https://zh-tw.coursera.org/learn/ntumlone-mathematicalfoundations): Taught by National Taiwan University Prof. Hsuan-Tien Lin 
