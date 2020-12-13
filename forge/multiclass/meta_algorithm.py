import numpy as np

from ..base.meta_algorithms import MetaAlgorithm
from ..utils.data_utils import create_ova_data, create_ovo_data, check_data_consistent_length
from ..utils.initialize_utils import new_instance

class OneVsAll(MetaAlgorithm):
    def __init__(self, model=None, optimizer=None, metrics=None, **kwargs):
        self.clfs = {}      # classifiers
        self.model = model  # model of classifiers; should be ‘soft’ binary classifier
        self.model_config = {}
        self.compile(optimizer, metrics, **kwargs)

    def compile(self, optimizer=None, metrics=None, **kwargs):
        if optimizer is not None: self.model_config["optimizer"] = optimizer
        if metrics is not None: self.model_config["metrics"] = metrics
        self.model_config.update(kwargs)
    
    def fit(self, x_train, y_train, labels=None, **kwargs):
        check_data_consistent_length(x_train, y_train)

        # one-hot encode y_train data
        if (len(y_train.shape) == 1 or                              # vector (1-d array)
           (len(y_train.shape) == 2 and y_train.shape[1] == 1)):    # matrix (2-d array) with a single column                                        
            y_train, labels = create_ova_data(y_train)
        else:
            labels = labels 

        # One versus one process
        n_class = len(labels)
        for i in range(n_class):
            model = new_instance(self.model)
            using_config = {var:self.model_config.get(var, None) 
                                for var in model.compile.__code__.co_varnames 
                                if var != "self"
                            } 
            model.compile(**using_config)
            model.fit(x_train, y_train[:, i], **kwargs)    # create classifiers by one-vs-all data
            self.clfs[labels[i]] = model
    
    def predict(self, x_predict):
        def get_pred_prob(x):
            return {clf.predict(x, result="prob")[0]:label for label, clf in self.clfs.items()}
            # np.ndarray is not hashable; get the result by indexing the first element
            # (and there is only one element)

        def get_max_prob_label(x):
            probs = get_pred_prob(x)
            return probs[max(list(probs))]

        y_predict = np.zeros(len(x_predict))
        for i in range(len(x_predict)):
            y_predict[i] = get_max_prob_label(x_predict[i, None])
            # x should be 2d thus index it by [:, None]
        return y_predict
        
    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.metrics.eval(y_test, y_pred)



class OneVsOne(MetaAlgorithm):

    def __init__(self, model=None, optimizer=None, metrics=None, **kwargs):
        self.clfs = {}      # classifiers
        self.model = model  # model of classifiers
        self.model_config = {}
        self.compile(optimizer, metrics, **kwargs)


    def compile(self, optimizer=None, metrics=None, **kwargs):
        if optimizer is not None: self.model_config["optimizer"] = optimizer
        if metrics is not None: self.model_config["metrics"] = metrics
        self.model_config.update(kwargs)


    def fit(self, x_train=None, y_train=None, label_x_y_dict=None, **kwargs):
        # create a dictionary whose labels(keys) are combinations of 2 values in y_train 
        #  and values are corresponding subsets of x_train and one-hot-encoded subset of
        #  y_train. If the dictionary is given as argument then skip the procedure.
        if label_x_y_dict is None:
            label_x_y_dict = create_ovo_data(x_train, y_train)

        # train each classifiers using one-vs-one data
        for i, data_train in label_x_y_dict.items():
            x_train, y_train = data_train
            model = new_instance(self.model)
            using_config = {var:self.model_config.get(var, None) 
                                for var in model.compile.__code__.co_varnames 
                                if var != "self"
                            } 
            model.compile(**using_config)
            model.fit(x_train, y_train, **kwargs)
            self.clfs[i] = model


    def predict(self, x_predict):
        predict_args = {"x_predict":x_predict}
        if "result" in self.model.predict.__code__.co_varnames:
            predict_args["result"]  = "class"       
        
        polls = np.empty((len(x_predict), len(self.clfs)), dtype=np.int32)
        i = 0
        for label, model in self.clfs.items():
            res = model.predict(**predict_args).astype(int)
            polls[:, i] = np.where(res==1, label[0], label[1])
            i = i + 1
        y_predict = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, polls)
        return y_predict
    
    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.metrics.eval(y_test, y_pred)