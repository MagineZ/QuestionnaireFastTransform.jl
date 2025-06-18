import datetime
import affinity
import dual_affinity
import bin_tree_build
import flex_tree_build
import numpy as np
import tree_util
import scipy.spatial as spsp
import collections
import transform
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import Mytree as tree
from Mytree import ClusterTreeNode
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import time
import numba
@numba.njit(parallel=True)

def fast_cityblock(ref, dat):
    N, D = ref.shape        
    M = dat.shape[0]        
    result = np.empty((N, M))
    for i in numba.prange(N):         # Parallelize over reference vectors
        for j in range(M):
            s = 0.0
            for d in range(D):        # Accumulate absolute differences
                s += abs(ref[i, d] - dat[j, d])
            result[i, j] = s
    return result

def bin_tree_build_roseland(X,cut_type="r_dyadic",bal_constant=1.0, method = 'euclidean',alpha = 1.0,beta = 0.0,row_tree = None):
    """
    Takes a static, square, symmetric nxn affinity on n nodes and 
    applies the second eigenvector binary cut algorithm to it.
    cut_types currently supported are: 
    r_dyadic:   random dyadic; uniform distribution on the legal splits
                based on the balance constant.
    zero:       splits the eigenvector at zero, subject to the balance constant 
    """
    
    _,n = X.shape

    root = tree.ClusterTreeNode(range(n))
    queue = [root]

    while max([x.size for x in queue]) > 1:
        new_queue = []
        for node in queue:
            if node.size > 2:
                #cut it
                if cut_type == "zero":
                    cut = zero_eigen_cut(node,affinity)
                elif cut_type == "r_dyadic":
                    left,right = bal_cut(node.size,bal_constant)
                    cut = random_dyadic_cut_roseland(node, X, left, right, method,alpha,beta,row_tree)
                node.create_subclusters(cut)
            else:
                #make the singletons
                node.create_subclusters(np.arange(node.size))
            new_queue.extend(node.children)
        queue = new_queue

    root.make_index()                
    return root    

def random_dyadic_cut_roseland(node,X,left,right, method = 'euclidean',alpha = 1.0, beta = 0.0, row_tree = None):
    """
    Returns a randomized cut of the affinity matrix (cutting at zero) 
    corresponding to the elements in node, under the condition of bal_constant.
    """ 
    new_data = X[:,node.elements]
    _,n = new_data.shape
    if n > 100:
        vecs,s,_ = DMroseland_metric(new_data.T, 2,None,method,row_tree,alpha,beta)
    else:
        vecs,s = DM(new_data.T)
    eig = vecs[:,0]
    eig_sorted = eig.argsort().argsort()
    cut_loc = np.random.randint(left,right+1)
    labels = eig_sorted < cut_loc
    
    return labels
    
def DMroseland_metric(X, Dim = 2, m=None, method = 'euclidean',row_tree = None,alpha = 1.0, beta = 0.0):
    n, p = X.shape
    if m is None:
        m = int(np.floor(np.sqrt(n)))
    step = n // m
    refdex = np.arange(step, n + 1, step) - 1
    ref = X[refdex, :]
    m = ref.shape[0]
    NN = m
    # Step 2: Construct affinity_ext and bandwidth
    if method == 'euclidean':
        affinity_ext = cdist(X, ref, metric='euclidean')
    elif method == 'emd1':
        affinity_ext = calc_emd_ref4(ref.T,X.T,row_tree,alpha,beta)
        affinity_ext = affinity_ext.T
        
    
    nbrs = NearestNeighbors(n_neighbors=NN).fit(ref)
    dista, _ = nbrs.kneighbors(X)
    bandw = dista[:, -1]

    # Step 3: Truncate large distances
    isocount = 0
    for i in range(n):
        close = affinity_ext[i, :] < 3 * bandw[i]
        if np.sum(close) <= 2:
            trunc = np.quantile(affinity_ext[i, :], 0.3)
            affinity_ext[i, affinity_ext[i, :] >= trunc] = np.inf
            isocount += 1
        else:
            affinity_ext[i, ~close] = np.inf

    q = 100 * np.sum(np.isinf(affinity_ext)) / (m * n)
    numref = np.sum(~np.isinf(affinity_ext), axis=1)
    if isocount / n > np.sqrt(n) / n:
        print("(warning) Too many points are not close to reference data.")

    # Step 4: Construct affinity matrix
    B = np.outer(bandw, bandw[refdex])
    W_ref = np.exp(-np.square(affinity_ext) / B)
    W_ref[np.isinf(W_ref)] = 0
    W_ref = csr_matrix(W_ref)

    D = W_ref @ np.asarray(W_ref.sum(axis=0)).flatten()
    V2 = 1.0 / np.sqrt(D + 1e-12)
    V2 = csr_matrix(np.diag(V2))
    W_tilde = V2 @ W_ref

    UD, S, _ = svds(W_tilde, k=Dim + 1)
    U = V2 @ UD
    S = S ** 2
    idx = np.argsort(S)[::-1]  # indices of S in descending order
    S = S[idx]
    U = U[:, idx]
    return U[:,1:], S[1:], W_tilde

    
def calc_emd_ref4(ref_data,data,row_tree,alpha=1.0,beta=0.0,weights=None):
    """
    Calculates the EMD from a set of points to a reference set of points
    The columns of ref_data are each a reference set point.
    The columns of data are each a point outside the reference set.
    """
    ref_rows,ref_cols = np.shape(ref_data)
    rows,cols = np.shape(data)
    assert rows == row_tree.size, "Tree size must match # rows in data."
    assert ref_rows == rows, "Mismatched row #: reference and sample sets."

    emd = np.zeros([ref_cols,cols])
    
    averages_mat = transform.tree_averages_mat(row_tree)
    ref_coefs = averages_mat.dot(ref_data)
    coefs = averages_mat.dot(data)
    
    folder_fraction = np.array([((node.size*1.0/rows)**beta)*
                                (2.0**((1.0-node.level)*alpha))
                                 for node in row_tree])
    if weights is not None:
        folder_fraction = folder_fraction*weights
    
    #coefs = np.diag(folder_fraction).dot(coefs)
    coefs *= folder_fraction[:, np.newaxis]
    #ref_coefs = np.diag(folder_fraction).dot(ref_coefs)   
    ref_coefs *= folder_fraction[:, np.newaxis]
    #emd = spsp.distance.cdist(ref_coefs.T,coefs.T,"cityblock")
    print("start emd")
    emd = fast_cityblock(ref_coefs.T, coefs.T)
    #ref_coefs = ref_coefs.T      # shape (90, 16383)
    #coefs = coefs.T          # shape (8192, 16383)
    # Now compute pairwise L1 (cityblock) distances
    #emd = np.abs(ref_coefs[:, np.newaxis, :] - coefs[np.newaxis, :, :]).sum(axis=2)
    return emd

def flex_tree_diffusion_roseland(data,penalty_constant,n_eigs=12,method = 'euclidean',alpha = 1.0,beta = 0.0,row_tree = None):
    """
    affinity is an nxn affinity matrix.
    Creates a flexible tree by calculating the diffusion on the given affinity.
    Then clusters at each level by the flexible tree algorithm. For each level
    up, doubles the diffusion time.
    penalty_constant is the multiplier of the median diffusion distance.
    """
    #First, we calculate the first n eigenvectors and eigenvalues of the 
    #diffusion
    cluster_list = []
    _,n = data.shape
    if n > 100:
        vecs,vals,_ = DMroseland_metric(data.T,n_eigs,None,method,row_tree,alpha,beta)
    else:
        vecs,vals = DM(data.T,n_eigs,method,row_tree,alpha,beta)
    diff_time = 1.0
    q = np.eye(data.shape[0])
    while 1:
        #now we calculate the diffusion distances between points at the 
        #current diffusion time.
        diff_vecs = vecs.dot(np.diag(vals**diff_time)) 
        diff_dists = spsp.distance.squareform(spsp.distance.pdist(diff_vecs))
        #we take the affinity between clusters to be the average diffusion 
        #distance between them.
        avg_dists = q.dot(diff_dists).dot(q.T)
        #now we cluster the points based on this distance
        cluster_list.append(cluster_from_distance(avg_dists,penalty_constant))
        #if there is only one node left, then we are done.
        #otherwise, add another level to the tree, double the diffusion time
        #and keep going.
        if len(cluster_list[-1]) == 1:
            break
        temp_tree = clusterlist_to_tree(cluster_list)
        cpart = ClusteringPartition([x.elements for x in temp_tree.dfs_level(2)])
        q,_ = cluster_transform_matrices(cpart)
        diff_time *= 2.0
    return clusterlist_to_tree(cluster_list)

def bal_cut(n,balance_constant):
    """
    Given n nodes and a balance_constant, returns the endpoints of the 
    interval of legal cutpoints for a binary tree.
    """ 
    if n==1:
        return 0,1
    left = int(np.ceil((1.0/(1.0+balance_constant))*n))
    right = int(np.floor((balance_constant/(1.0+balance_constant))*n))
    if left > right and n % 2 == 1:
        left = int(np.floor(n/2.0))
        right = int(np.ceil(n/2.0))
    elif left > right:
        left = right
    return left,right    


def DM(X, epsilon=None, n_eigs=10,Dim = 2,method = 'euclidean',row_tree = None,alpha = 1.0, beta = 0.0):
    """
    Compute diffusion map of data matrix X ∈ ℝ^{n×d}
    
    Parameters:
        X       : ndarray, shape (n_samples, n_features)
        epsilon : float, Gaussian kernel bandwidth. If None, use median heuristic.
        n_eigs  : int, number of nontrivial eigenvectors to return
        t       : int, diffusion time
    
    Returns:
        lambdas : ndarray, shape (n_eigs,)
        psi     : ndarray, shape (n_samples, n_eigs)
    """
    n = X.shape[0]

    # Step 1: pairwise distance matrix
    if method == 'euclidean':
        D = squareform(pdist(X, metric='euclidean'))
    elif method == 'emd1':
        #affinity_ext = calc_emd_ref4(ref.T,X.T,row_tree,alpha,beta)
        #affinity_ext = affinity_ext.T
        D = dual_affinity.calc_emd(X.T,row_tree,alpha,beta)

    # Step 2: Gaussian affinity
    if epsilon is None:
        epsilon = np.median(D)
    W = np.exp(-D**2 / (4 * epsilon))

    # Step 3: Markov normalization
    d = np.sum(W, axis=1)
    P = W / d[:, None]  # row-normalize

    # Step 4: Eigen-decomposition
    eigvals, eigvecs = eigh(P)

    # Step 5: Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 6: Skip the trivial first eigenvector (λ ≈ 1)
    return eigvecs[:, 1:n_eigs+1],eigvals[1:n_eigs+1]

def DMroseland_cosine_metric(X, Dim = 2,row_tree = None,alpha = 1.0, removmean = 1.0): 
    W_ref = dual_affinity.partition_dualgeometry_ref(np.expand_dims(X, axis=2), row_tree, alpha, removmean)
    W_ref = W_ref.T
    W_ref[np.isinf(W_ref)] = 0
    W_ref = csr_matrix(W_ref)

    D = W_ref @ np.asarray(W_ref.sum(axis=0)).flatten()
    V2 = 1.0 / np.sqrt(D + 1e-12)
    V2 = csr_matrix(np.diag(V2))
    W_tilde = V2 @ W_ref

    UD, S, _ = svds(W_tilde, k=Dim + 1)
    U = V2 @ UD
    S = S ** 2
    idx = np.argsort(S)[::-1]  # indices of S in descending order
    S = S[idx]
    U = U[:, idx]
    return U[:,1:], S[1:], W_tilde