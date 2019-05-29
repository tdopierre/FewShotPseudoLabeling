from scipy import cluster
import numpy as np


def get_unique_label_trees(root_tree: cluster.hierarchy.ClusterNode, labels, max_dist=None, path='root'):
    if max_dist is None:
        max_dist = np.inf

    if root_tree.is_leaf():
        return [(root_tree, path)]
    found_labels = [labels[ix] for ix in root_tree.pre_order() if labels[ix] != '']

    # Case when tree contains only unlabelled samples
    if len(found_labels) == 0:
        if root_tree.dist < max_dist:
            return [(root_tree, path)]

    # Case when tree contains at most 1 unique label
    elif len(set(found_labels)) == 1:
        if root_tree.dist < max_dist:
            return [(root_tree, path)]

    # Fallback case: more than 2 unique labels in the tree, or distance objective not reached
    return (get_unique_label_trees(root_tree.left, labels, max_dist, path=f'{path}.left') +
            get_unique_label_trees(root_tree.right, labels, max_dist, path=f'{path}.right'))
