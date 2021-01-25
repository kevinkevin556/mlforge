import numpy as np


class Node():
    def __init__(self):
        # Decision
        self.feature_ = None
        self.threshold_ = None
        self.leaf_prediction_ = None   # value or function
        
        # Tree
        self.is_leaf_ = None
        self.is_root_ = None
        self.parent_ = None
        self.children_ = {}

        # Random subspace (in Random Forest)
        self.feature_projection_ = None
    
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if self.is_leaf_:
            if callable(self.leaf_prediction_):
                return self.leaf_prediction_(X)
            else:
                return self.leaf_prediction_
        else:
            output = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                if self.threshold_:
                    label = "left" if X[i, self.feature_] <= self.threshold_ else "right"
                else:
                    label = X[i, self.feature_]

                output[i] = self.children_[label].predict(X[i, :])
            return output



def parse_tree(tree, node_info=[], label="Root"):
    
    node_annotation = (label + ": " + "{}").format(repr(tree))

    if len(tree.children_) == 0:
        output = [(node_info, node_annotation)]
    else:
        keys = list(tree.children_.keys())
        output = [(node_info, node_annotation)]
        for i in range(len(tree.children_)):
            if i == len(tree.children_)-1:
                child_info = node_info + [-1]
            else:
                child_info = node_info + [i]
            for info in parse_tree(tree.children_[keys[i]], node_info=child_info, label=str(keys[i])):
                output.append(info)
    return output


def parse_hierarchy(hierarchy):
    output = ""
    for i in range(len(hierarchy)):
        if i == len(hierarchy)-1:
            if hierarchy[i] == -1:
                output += "└────"
            else:
                output += "├────"
        else:
            if hierarchy[i] == -1:
                output += "      "
            else:
                output += "│     " 
    return output


def print_tree(tree):
    parsed_tree = parse_tree(tree)
    for i in range(len(parsed_tree)):
        node = parsed_tree[i]
        if i == 0:
            print(parse_hierarchy(node[0]) + node[1])
        else:
            print(parse_hierarchy(node[0]) + " " + node[1])