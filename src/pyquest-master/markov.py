"""
markov.py: Function to calculate vectors and eigenvalues of Markov chain
           based on an affinity.
"""

import numpy as np
import scipy.sparse.linalg as spsl
import scipy.sparse as sps

#### sparse matrices untested  #####

def make_markov_symmetric(data,thres=1e-8):
    """
    data is a (symmetric) affinity matrix. elements less than thres are zeroed.
    Returns a "symmetrized" and normalized version of the Markov chain matrix. 
    """    
    
    
    if sps.issparse(data) == True:
        d_mat = data.multiply(data > thres)
        rowsums = 1.0/(d_mat.sum(axis=1) + 1e-15)
        m,n =  np.shape(d_mat)
        invD = sps.spdiags(rowsums.T, 0,m,n)
        p_mat = invD * d_mat * invD
        rowsums = np.sqrt(1.0/(p_mat.sum(axis=1) + 1e-15))
        sqrtInvD = sps.spdiags(rowsums.T, 0, m,n)
        p_mat = sqrtInvD * d_mat * sqrtInvD
    else:
        d_mat = data*(data > thres)
        rowsums = np.sum(d_mat,axis=1) + 1e-15
        p_mat = d_mat/(np.outer(rowsums,rowsums))
        d_mat2 = np.sqrt(np.sum(p_mat,axis=1)) + 1e-15
        p_mat = p_mat/(np.outer(d_mat2,d_mat2))
    return p_mat

def make_markov_row_stoch(data,thres=1e-8):
    """
    data is a (symmetric) affinity matrix. elements less than thres are zeroed.
    Returns the row stochastic Markov matrix. 
    """    
    
    
    if sps.issparse(data) == True:
        d_mat = data.multiply(data > thres)
        m,n =  np.shape(d_mat)
        rowsums = 1.0/(d_mat.sum(axis=1) + 1e-15)
        invD = sps.spdiags(rowsums.T, 0, m,n)
        p_mat = invD * d_mat
    else:
        d_mat = data*(data > thres)
        rowsums = 1.0/(np.sum(d_mat,axis=1) + 1e-15)
        p_mat = np.diag(rowsums).dot(d_mat)
    return p_mat

def markov_eigs(data,n_eigs,normalize=True,thres=1e-8):
    """
    data is a (symmetric) affinity matrix.
    n_eigs is the number of eigenvalues/eigenvectors desired.
    normalize sets whether to normalize all the eigenvectors such that the first
    eigenvector is 1.   (this function is ncut from the MATLAB questionnaire)
    Returns the first n eigenvectors and the corresponding eigenvalues.
    """
    p_mat = make_markov_symmetric(data,thres)
    return _calc_eigs(p_mat,n_eigs,normalize,thres)

def _calc_eigs(markov_chain,n_eigs,normalize=True,thres=1e-8):
    n = np.shape(markov_chain)[0]
    n_eigs = min(n_eigs,n)
    [vectors,singvals,_] = spsl.svds(markov_chain,n_eigs,random_state=1357)
    y = np.argsort(-singvals)
    eigenvalues = singvals[y]
    eigenvectors = vectors[:,y]
    
    if normalize:
        n_mat = np.hstack([np.reshape([eigenvectors[:,0]],[-1,1])]*n_eigs)
        eigenvectors /= n_mat
        n_mat2 = np.vstack([np.sign(eigenvectors[0,1:])]*n)
        n_mat2[n_mat2==0] = 1.0
        eigenvectors[:,1:] *= n_mat2
        
    return eigenvectors, eigenvalues
    