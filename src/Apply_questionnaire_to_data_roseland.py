import sys
import os

# Add the directory containing Packages.py to the Python path
sys.path.append(os.path.abspath('./pyquest-master/'))
#sys.path.append("/CosMx/CosMx/Julia_allcodes_plus_dependencies/pyquest-master")
from imports import *
import questionnaire_roseland2 as qcoif
from scipy.spatial.distance import squareform, pdist
import numpy as np
import affinity
import networkx as nx

def main(A,switchver=False,switchcos = 1):

    ### Now for Raphy's questionnaire method
    kwargs = {}
    kwargs["threshold"] = 0.0
    kwargs["row_alpha"] = 0.0
    kwargs["col_alpha"] = 0.0
    kwargs["row_beta"] = 1.0
    kwargs["col_beta"] = 1.0
    kwargs["tree_constant"] = 1.0
    kwargs["n_iters"] = 3
    kwargs["knn"] = int(np.ceil(min(np.shape(A))**(1/2)))
    kwargs["epsilon"] = 1

    if switchcos == 1:
        params = qcoif.PyQuestParams(qcoif.INIT_AFF_COS_SIM,
                                      qcoif.TREE_TYPE_FLEXIBLE,
                                      qcoif.DUAL_COSINE,
                                      qcoif.DUAL_COSINE, **kwargs)
       
    elif switchcos == 2: 
        #init_row_aff = affinity.mutual_cosine_similarity(A.T, threshold=0.0)
        #init_col_aff = affinity.mutual_cosine_similarity(A, threshold=0.0)
        #params = qcoif.PyQuestParams(qcoif.INIT_AFF_COS_SIM,
        #                              qcoif.TREE_TYPE_FLEXIBLE,
        #                              qcoif.DUAL_COSINE,
        #                              qcoif.DUAL_COSINE, **kwargs)
        params = qcoif.PyQuestParams(qcoif.INIT_AFF_GAUSSIAN,
                                      qcoif.TREE_TYPE_FLEXIBLE,
                                      qcoif.DUAL_EMD,
                                      qcoif.DUAL_COSINE, **kwargs)
    else:
        params = qcoif.PyQuestParams(qcoif.INIT_AFF_GAUSSIAN,
                                      qcoif.TREE_TYPE_BINARY,
                                      qcoif.DUAL_EMD,
                                      qcoif.DUAL_EMD, **kwargs)

    #init_row_vecs, init_row_vals = markov.markov_eigs(init_row_aff, 12)
    #init_col_vecs, init_col_vals = markov.markov_eigs(init_col_aff, 12)
   
    qrun = qcoif.pyquest(A, params)
    order_row = np.array(mst_ordering(qrun.row_aff))
    order_col = np.array(mst_ordering(qrun.col_aff))
    
    return qrun,order_row,order_col


def mst_ordering(W):
    """Finds an ordering using Minimum Spanning Tree traversal."""
    # Convert affinity matrix to distance matrix (inverted similarity)
    D = np.max(W) - W  # Higher similarity = lower distance
    
    # Construct a graph and compute the Minimum Spanning Tree (MST)
    G = nx.from_numpy_array(D)
    T = nx.minimum_spanning_tree(G)

    # Perform Depth-First Search (DFS) to get an ordering
    order = list(nx.dfs_preorder_nodes(T, 0))  # Start from node 0
    return order
