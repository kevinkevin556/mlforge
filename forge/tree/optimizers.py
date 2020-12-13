import numpy as np

from ..base.optimizers import Optimizer
from ..utils.initialize_utils import init_weight, set_x_train, set_y_train
from ..impurities import MeanSquaredError, GiniIndex, Entropy
from ..utils.decorator_utils import implementation
from .node import Node


class DecisionStumpSolver(Optimizer):
    def __init__(self):
        pass

    def execute(self, X, y):
        x = set_x_train(X, add_bias=False)
        y = set_y_train(y)

        optimal_sign, feature, threshold = self.solve_split(x, y)
        return optimal_sign, feature, threshold


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="recursive_solution")
    def solve_split(self, x, y):
        split = None
        optimal_sign, feature, threshold = None, None, None
        error = None
        min_error = x.shape[0]
        
        # d iterations, each iteration with O(N*logN) => O(d * NlogN) time
        for i in range(x.shape[1]): 
            sorted_id = np.argsort(x[:, i], kind='stable') # O(N*logN)
            xi_sorted = x[sorted_id, i]
            y_sorted = y[sorted_id]
            
            positive_ray, negative_ray, _ = self.recursively_solve(y_sorted, len(y_sorted)) # O(N)
            if positive_ray["error"] < negative_ray["error"]:
                split = positive_ray
            else:
                split = negative_ray
                
            if split["error"] < min_error:
                min_error = split["error"]
                feature = i
                threshold = self.find_threshold(xi_sorted, split["position"])
                optimal_sign = split["sign"]

        return optimal_sign, feature, threshold


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="recursive_solution")
    def find_threshold(self, xi_sorted, pos):
        if pos == 0:
            threshold = xi_sorted[0] - 1
        elif pos == len(xi_sorted):
            threshold = xi_sorted[-1] + 1
        else:
            threshold = 0.5 * (xi_sorted[pos-1] + xi_sorted[pos])
        return threshold


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="recursive_solution")
    def recursively_solve(self, partition, partition_length):
        """ Solve the problem in divide and conquer approach.

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
        positive_ray, negative_ray, counts = None, None, None

        if partition_length == 1:            
            if partition[0] == 1:
                #        split = [sign, position, error]
                positive_ray = {"sign":  1, "position": 0, "error": 0}
                negative_ray = {"sign": -1, "position": 1, "error": 0}
                counts = {+1: 1, -1: 0}
            
            if partition[0] == -1:
                positive_ray = {"sign":  1, "position": 1, "error": 0}
                negative_ray = {"sign":  -1, "position": 0, "error": 0}
                counts = {+1: 0, -1: 1}
        else: 
            # Divide
            midpoint = partition_length // 2
            front_partition, back_partition = partition[0:midpoint], partition[midpoint:partition_length]
            
            # Conquer
            front_pos_ray, front_neg_ray, front_counts = \
                self.recursively_solve(front_partition, midpoint)
            back_pos_ray, back_neg_ray, back_counts = \
                self.recursively_solve(back_partition, partition_length-midpoint)
            
            positive_ray = \
                self.find_best_split(1, front_pos_ray, front_counts, back_pos_ray, back_counts, midpoint)
            negative_ray = \
                self.find_best_split(-1, front_neg_ray, front_counts, back_neg_ray, back_counts, midpoint) 

            counts = {
                +1: front_counts[+1] + back_counts[+1],
                -1: front_counts[-1] + back_counts[-1]
            }
        
        return positive_ray, negative_ray, counts


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="recursive_solution")
    @staticmethod
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
            output["error"] = error_applying_back_split

        return output



class CART(Optimizer):
    def __init__(self, criterion=None, max_height=np.inf):
        self.criterion = criterion
        self.problem  = "regression" if type(criterion) is MeanSquaredError else "classification"
        self.max_height = max_height

    def execute(self, X, y):
        x = set_x_train(X, add_bias=False)
        y = set_y_train(y)
        
        root = self.build_tree(x, y)
        root.is_root_ = True
        
        return root

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def build_tree(self, X, y, current_height=0):
        node = Node()

        if self.is_terminated(X, y, current_height):                        
            node = self.create_leaf_node(node, y)
        else:
            node = self.create_sub_trees(node, X, y, current_height)

        return node

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def is_terminated(self, X, y, current_height):
        identical_ys = (len(np.unique(y)) == 1)
        identical_xs = np.apply_along_axis(arr=X, func1d=lambda x:len(np.unique(x))==1, axis=0).all()
        meet_height_limit = current_height == self.max_height
        
        is_terminated = (identical_xs or identical_ys or meet_height_limit)
        return is_terminated


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def create_leaf_node(self, node, y):
        node.is_leaf_ = True

        # prediction of CART are constant leaves which depend on type of problem
        if self.problem == "regression":
            node.leaf_prediction_ = np.mean(y)
        if self.problem == "classification":
            y_values, counts = np.unique(y, return_counts=True)
            node.leaf_prediction_ = y_values[counts.argmax()]
        
        return node


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def create_sub_trees(self, node, X, y, current_height):
        # learn the best decision stump and split the data into 2 parts
        # left: X[n, feature] <= theta,  right: X[n, feature] > theta
        node.feature_, branches, node.threshold_ = self.solve_branches(X, y)

        for side in ["left", "right"]:
            child_node = self.build_tree(X[branches[side], :], y[branches[side]], current_height+1)
            node.children_[side] = child_node
        
        return node


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def solve_branches(self, X, y):
        min_impur = np.inf
        feature = None
        threshold = None

        for i in range(X.shape[1]):
            sort_id = np.argsort(X[:, i])
            xi_sorted= X[sort_id, i]
            yi_sorted = y[sort_id]

            position, impur = self.criterion.find_split(yi_sorted)
            
            if impur < min_impur:
                min_impur = impur
                feature = i
                if position == 0 :
                    threshold = xi_sorted[0] - 1
                elif position == len(xi_sorted):
                    threshold = xi_sorted[-1] + 1
                else:
                    threshold = 0.5 * (xi_sorted[position-1] + xi_sorted[position])
        
        branches = {
            "left":  np.where(X[:, feature] <= threshold)[0],
            "right": np.where(X[:, feature] > threshold)[0]
        }
        return feature, branches, threshold



class ID3(Optimizer):
    def __init__(self, criterion=Entropy()):
        self.criterion = criterion
        self.problem = "classification"


    def execute(self, X, y):
        """ In ID3 decision tree, X should be categorical input."""
        x = set_x_train(X, add_bias=False)
        y = set_y_train(y)
        
        root = self.build_tree(x, y)
        root.is_root_ = True
        
        return root


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def build_tree(self, X, y, features_remained=None):
        node = Node()
        features_remained = list(range(X.shape[1])) if not features_remained else features_remained

        if self.is_terminated(X, y, features_remained):                        
            node = self.create_leaf_node(node, y)
        else:
            node = self.create_sub_trees(node, X, y, features_remained)
        
        return node
                
    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def is_terminated(self, X, y, features_remained):
        no_feature_left = (len(features_remained) == 0)
        identical_ys = (len(np.unique(y)) == 1)
        identical_xs = np.apply_along_axis(
                        arr = X[:, features_remained],
                        func1d = lambda x:len(np.unique(x))==1,
                        axis = 0
                    ).all() if not no_feature_left else False
        
        is_terminated = (identical_xs or identical_ys or no_feature_left)
        return is_terminated

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def create_leaf_node(self, node, y):
        node.is_leaf_ = True
        y_values, counts = np.unique(y, return_counts=True)
        node.leaf_prediction_ = y_values[counts.argmax()]
        return node

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def create_sub_trees(self, node, X, y, features_remained):
        feature_chosen_id, branches = self.solve_branches(X[:, features_remained], y)
        feature_chosen = features_remained[feature_chosen_id]
        node.feature_ = feature_chosen
        features_remained = list(set(features_remained) - set([feature_chosen]))

        for feature_value, x_id in branches.items():
            child_node = self.build_tree(X[x_id, :], y[x_id], features_remained)
            child_node.parent_ = node
            node.children_[feature_value] = child_node
        
        return node

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build tree")
    def solve_branches(self, X, y):
        n = len(X)
        init_entropy = self.criterion.eval(y)
        max_information_gain = -1
        feature = None

        for i in range(X.shape[1]):
            entropy_sum = 0
            for value in np.unique(X[:,i]):
                count = np.sum(X[:, i]==value)
                entropy_sum += count/n * self.criterion.eval(y[X[:, i]==value])
            
            information_gain = init_entropy - entropy_sum
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                feature = i
        
        branches = {i: np.where(X[:, feature] == i)[0] for i in np.unique(X[:, feature])}
        return feature, branches
 


class RandomTree(Optimizer):

    def __init__(self, criterion=None, tree=CART(), random_subspace=(0.5, ), **kwargs):
        self.random_subspace = random_subspace
        
        if criterion:
            self.criterion = criterion
            self.tree = tree.__class__(criterion=criterion, **kwargs)
        else:
            self.criterion = tree.criterion
            self.tree = tree

        self.tree.build_tree = self.randomize(self.tree.build_tree)


    def execute(self, X, y):
        x = set_x_train(X, add_bias=False)
        y = set_y_train(y)

        root = self.tree.build_tree(x, y)
        root.is_root_ = True

        return root

    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build random tree")
    def randomize(self, build_tree):
        def randomized_build_tree(X, y, *args, **kwargs):
            feature_projection = self.randomize_features(X.shape[1], self.random_subspace)
            x = X @ feature_projection
            
            node = build_tree(x, y, *args, **kwargs)
            node.feature_projection_ = feature_projection
            return node

        return randomized_build_tree


    @implementation(numba_jit=False, multiple_func_impl=True, func_group="build random tree")
    def randomize_features(self, n_feature, random_subspace):
        def set_sample_num(n, total):
            if type(n) is float and n > 0 and n < 1:
                return np.max([1, np.round(n * total).astype(int)])
            elif type(n) is int and n > 0:
                return np.min([n, total])
            else:
                raise ValueError("Invalid configuration for feature randomization.")

        # Initialize variables
        if random_subspace is None or len(random_subspace) == 0:
            mode = "no randomize"
            n_selected = 0
            n_generated = 0
        elif len(random_subspace)==1:
            mode = "forest-RI"
            n_selected = set_sample_num(random_subspace[0], n_feature)     # number of sampled features
            n_generated = n_selected 
        elif len(random_subspace) == 2:
            mode = "forest-RC"
            n_selected = set_sample_num(random_subspace[0], n_feature)     # number of sampled features 
            n_generated = set_sample_num(random_subspace[1], n_feature)    # number of newly-generated features 
        else:
            raise ValueError("Invalid random subspace.")

        # Random procedures:
        transformation = np.zeros([n_feature, n_generated])
        selected_feature_index = np.random.choice(range(n_feature), size=n_selected, replace=False)

            # random input variable selection (forest-RI)
        if mode == "forest-RI": 
            transformation[selected_feature_index, np.arange(n_generated)] = 1

            # linear combinations of input variables (forest-RC)
        elif mode == "forest-RC": 
            combination_weight = np.random.uniform(low=-1.0, high=1.0, size=(n_selected, n_generated))
            transformation[selected_feature_index.reshape(-1,1), np.arange(n_generated)] = combination_weight 

        elif mode == "no randomize":
            transformation = np.identity(n_feature)

        return transformation