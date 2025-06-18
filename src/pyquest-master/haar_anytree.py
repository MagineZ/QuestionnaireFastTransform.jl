"""
haar.py: Functions for describing Haar, Haarlike, and tensor Haar bases on
         discrete spaces and products of discrete spaces.
"""

import numpy as np
import scipy.linalg
from anytree import PreOrderIter, LevelGroupOrderIter, LevelOrderIter
from anytree import Node, util, findall_by_attr

def haar_vectors(n, node_sizes, norm="L2"):
    """
    Returns a matrix of haar basis vectors for a tree with n subnodes of
    sizes given in node_sizes.
    norm should be either "L2" or "L1"; L2 gives an orthonormal basis
    while L1 shifts the normalization to the inverse transform.
    """

    haar_basis = np.zeros([n, n])
    haar_basis[:, 0] = 1.0

    # if we want the orthonormal basis, then we normalize by 1/sqrt(n) on the
    # constant vector.
    if norm == "L2":
        haar_basis[:, 0] /= np.sqrt(n)

    for i in range(1, n):
        pluses = node_sizes[i - 1]
        minuses = np.sum(node_sizes[i:])

        haar_basis[i - 1, i] = minuses
        haar_basis[i:, i] = -1.0 * pluses

        if norm == "L2":
            norm_i = np.sqrt(np.sum((haar_basis[:, i] ** 2) * (node_sizes)))
        elif norm == "L1":
            norm_i = 2.0 * (minuses * pluses) / (minuses + pluses)

        haar_basis[:, i] /= norm_i

    return haar_basis


def compute_haar(t, return_nodes=False, norm="L2"):
    """
    Takes a full tree of type ClusterTreeNode and computes the canonical
    Haar-like basis.
    return_nodes specifies whether we want to return parent node.idx associated
    with each basis vector. (-1 means this is the root ie constant vector)
    """

    n = t.size
    haar_basis = np.zeros([n, n])
    node_ids = np.zeros(n, int)
    node_ids[0] = -1
    cur_col = 1
    haar_basis[:, 0] = 1.0
    if norm == "L2":
        haar_basis[:, 0] /= np.sqrt(n)

    for node in t:
        node_size = len(node.children)
        schildren = list(reversed(sorted(node.children, key=lambda x: x.size)))
        if node_size > 0:
            basis_vecs = haar_vectors(node_size,
                                      np.array([x.size for x in schildren]),
                                      norm)

            for i in range(1, node_size):  # ignores node with 1 child
                # each basis vector will be a column of the basis
                for j, child in enumerate(schildren):
                    haar_basis[child.elements, cur_col] = basis_vecs[j, i]
                    node_ids[cur_col] = node.idx
                cur_col += 1

    if return_nodes:
        return haar_basis, node_ids
    else:
        return haar_basis

def compute_averagingbasis(t, return_nodes=False, norm="L2"):
    """
    Takes a full tree of type ClusterTreeNode and computes the canonical
    Haar-like basis.
    return_nodes specifies whether we want to return parent node.idx associated
    with each basis vector. (-1 means this is the root ie constant vector)
    """
    counterone=0
    n = t.size
    haar_basis = np.zeros([n, n])
    node_ids = np.zeros(n, int)
    node_ids[0] = -1
    cur_col = 1
    haar_basis[:, 0] = 1.0
    if norm == "L2":
        haar_basis[:, 0] /= np.sqrt(n)

    for node in t:
        node_size = len(node.children)
        schildren = list(reversed(sorted(node.children, key=lambda x: x.size)))
        if node_size > 0:
            basis_vecs = haar_vectors(node_size,
                                      np.array([x.size for x in schildren]),
                                      norm)
            #basis_vecs[:,0]/=scipy.linalg.norm(basis_vecs[:,0])*np.sqrt(np.array([x.size for x in schildren]))
            basis_vecs[:,0]/=np.sqrt(np.average([x.size for x in schildren]))
            if node_size>1:
                # each basis vector will be a column of the basis
                for j, child in enumerate(schildren):
                    haar_basis[child.elements, cur_col] = basis_vecs[j, 0]
                    #node_ids[cur_col] = node.idx
                node_ids[cur_col]=node.level
                newcol=cur_col+node_size-1
                node_ids[cur_col:newcol] = node_ids[cur_col]
                cur_col=newcol
            else:
                counterone+=1
    cur_col=0
    if counterone>0:
        leftover_basis = np.zeros([n, counterone])
        leftover_ids = np.zeros(counterone, int)
        for node in t:
            node_size = len(node.children)
            schildren = list(reversed(sorted(node.children, key=lambda x: x.size)))
            if node_size > 0:
                basis_vecs = haar_vectors(node_size,
                                          np.array([x.size for x in schildren]),
                                          norm)
                # basis_vecs[:,0]/=scipy.linalg.norm(basis_vecs[:,0])*np.sqrt(np.array([x.size for x in schildren]))
                basis_vecs[:, 0] /= np.sqrt(np.average([x.size for x in schildren]))
                if node_size==1:
                    child = node.children[0]
                    leftover_basis[child.elements, cur_col] = basis_vecs[0, 0]
                    leftover_ids[cur_col] = node.level
                    cur_col+=1

    if return_nodes:
            if counterone>0:
                return [haar_basis,leftover_basis], [node_ids,leftover_ids]
            else:
                return [haar_basis], [node_ids]
    else:
                return [haar_basis]



def compute_haar_anytree(root, return_nodes=False, norm="L2"):
    n = sum(1 for _ in PreOrderIter(root, filter_=lambda node: node.is_leaf))
    #if n <170:
    #    print('wat!')
    haar_basis = np.zeros([n, n])
    node_ids = np.zeros(n, int)
    node_ids[0] = -1  # Root node handling
    cur_col = 1
    haar_basis[:,0] = 1.0 / np.sqrt(n) if norm == "L2" else 1.0

    node_index_map = {node: int(node.name) for idx, node in enumerate(sorted(PreOrderIter(root), key=lambda x: len(x.leaves))) if not node.children}
#    for node in PreOrderIter(root):
    for node in LevelOrderIter(root):
        if node.children:
            node_size = len(node.children)
            #sizes = [child.size for child in node.children]
            #basis_vecs = haar_vectors(node_size, sizes, norm)
            # Further processing to assign vectors
            node_sizes = [len(child.leaves) or 1 for child in node.children]  # Using descendants or 1 if leaf
            basis_vecs = haar_vectors(node_size, np.array(node_sizes), norm)

            # Iterate over the vectors and assign them to the matrix
            for i in range(1, len(node.children)):  # Start from 1 to skip the constant vector
                for j, child in enumerate(node.children):
                    #child_index = node_index_map.get(child)
                    #if child_index is not None:
                        #haar_basis[child_index, cur_col] = basis_vecs[j, i]
                    try:
                        haar_basis[[int(x.name) for x in child.leaves], cur_col] = basis_vecs[j, i]
                    except:
                        print("i=", i)
                        print("j=",j)
                        print("current column=",cur_col)
                        print("basis_vecs shape",basis_vecs.shape)
                        print("haar_basis shape",haar_basis.shape)

                node_ids[cur_col] = int(node.name)
                cur_col += 1

    if return_nodes:
            return haar_basis, node_ids
    else:
            return haar_basis

def compute_averagingbasis_anytree(root, return_nodes=False, norm="L2"):
    n = sum(1 for _ in PreOrderIter(root, filter_=lambda node: node.is_leaf))
    counterone=0
    if n <170:
        print('wat!')
    haar_basis = np.zeros([n, n])
    node_ids = np.zeros(n, int)
    node_ids[0] = -1  # Root node handling
    cur_col = 1
    haar_basis[:,0] = 1.0 / np.sqrt(n) if norm == "L2" else 1.0

    node_index_map = {node: int(node.name) for idx, node in enumerate(sorted(PreOrderIter(root), key=lambda x: len(x.leaves))) if not node.children}
#    for node in PreOrderIter(root):
    for node in LevelOrderIter(root):
        if node.children:
            node_size = len(node.children)
            #sizes = [child.size for child in node.children]
            #basis_vecs = haar_vectors(node_size, sizes, norm)
            # Further processing to assign vectors
            node_sizes = [len(child.leaves) or 1 for child in node.children]  # Using descendants or 1 if leaf
            basis_vecs = haar_vectors(node_size, np.array(node_sizes), norm)
            basis_vecs[:, 0] /= np.sqrt(np.average(node_sizes))
            # Iterate over the vectors and assign them to the matrix
            # Start from 1 to skip the constant vector
            if node_size>1:
                for j, child in enumerate(node.children):
                    #child_index = node_index_map.get(child)
                    #if child_index is not None:
                        #haar_basis[child_index, cur_col] = basis_vecs[j, i]
                    try:
                        haar_basis[[int(x.name) for x in child.leaves], cur_col] = basis_vecs[j, 0]
                    except:
                        print("j=",j)
                        print("current column=",cur_col)
                        print("basis_vecs shape",basis_vecs.shape)
                        print("haar_basis shape",haar_basis.shape)
                node_ids[cur_col]=int(node.depth)
                newcol=cur_col+node_size-1
                node_ids[cur_col:newcol] = node_ids[cur_col]
                cur_col=newcol
            else:
                counterone+=1
    if counterone>0:
        leftover_basis = np.zeros([n, counterone])
        leftover_ids = np.zeros(counterone, int)
        cur_col = 0
        for node in LevelOrderIter(root):
            if node.children:
                node_size = len(node.children)
                # sizes = [child.size for child in node.children]
                # basis_vecs = haar_vectors(node_size, sizes, norm)
                # Further processing to assign vectors
                node_sizes = [len(child.leaves) or 1 for child in
                              node.children]  # Using descendants or 1 if leaf
                basis_vecs = haar_vectors(node_size, np.array(node_sizes), norm)
                basis_vecs[:, 0] /= np.sqrt(np.average(node_sizes))
                if node_size == 1:
                    child = node.children[0]
                    try:
                        haar_basis[[int(x.name) for x in child.leaves], cur_col] = basis_vecs[0, 0]
                    except:
                        print("current column=", cur_col)
                        print("basis_vecs shape", basis_vecs.shape)
                        print("haar_basis shape", haar_basis.shape)
                    leftover_ids[cur_col] = int(node.depth)
                    cur_col += 1

    if return_nodes:
            if counterone>0:
                return [haar_basis,leftover_basis], [node_ids,leftover_ids]
            else:
                return [haar_basis], [node_ids]
    else:
                return [haar_basis]



def build_long_BCR_transform_anytree(treeroot,level,norm="L2"):
    #reorder=np.argsort(compute_averagingbasis_anytree(treeroot,True)[1])
    hb=compute_haar_anytree(treeroot,False)#[:,reorder]
    sums,squence=compute_averagingbasis_anytree(treeroot,True)
    #sums=sums[:,reorder]
    #squence=squence[reorder]
    colfilter=scipy.linalg.norm(sums[0][:,np.where(squence[0]==level)[0]],axis=0)>0
    filtersums=sums[0][:,np.where(squence[0]==level)[0]][:,colfilter]
    hblevel=hb[:,np.where(squence[0]==level)[0]]
    if len(squence)>1:
        filtersums2 = sums[1][:, np.where(squence[1] == level)[0]]
        filtersums=np.hstack((filtersums,filtersums2))
    #rowfilter=np.where(scipy.linalg.norm(filtersums,axis=1)==0)[0]
    #if len(rowfilter)>0:
    #    toadd=np.eye(hb.shape[0])[:,rowfilter]
    #    finalstack=np.hstack((hb[:,np.where(squence==level)[0]],filtersums,toadd))
    #else:
    prestack = np.hstack((hblevel, filtersums))
    rowfilter=np.where(scipy.linalg.norm(prestack, axis=1) == 0)[0]
    if len(rowfilter)>0:
        toadd=np.eye(hb.shape[0])[:,rowfilter]
        finalstack=np.hstack((prestack,toadd))
    else:
        finalstack=prestack

    return hblevel.shape[1], prestack.shape[1], finalstack

def build_long_BCR_transform_CTN(treeroot,level,norm="L2"):
    #reorder=np.argsort(compute_averagingbasis(treeroot,True)[1])
    hb=compute_haar(treeroot,False)#[:,reorder]
    sums,squence=compute_averagingbasis(treeroot,True)
    #sums=sums[:,reorder]
    #squence=squence[reorder]
    colfilter=scipy.linalg.norm(sums[0][:,np.where(squence[0]==level)[0]],axis=0)>0
    filtersums=sums[0][:,np.where(squence[0]==level)[0]][:,colfilter]
    hblevel=hb[:,np.where(squence[0]==level)[0]]
    if len(squence)>1:
        filtersums2 = sums[1][:, np.where(squence[1] == level)[0]]
        filtersums=np.hstack((filtersums,filtersums2))
    prestack = np.hstack((hblevel, filtersums))
    rowfilter=np.where(scipy.linalg.norm(prestack, axis=1) == 0)[0]
    if len(rowfilter)>0:
        toadd=np.eye(hb.shape[0])[:,rowfilter]
        finalstack=np.hstack((prestack,toadd))
    else:
        finalstack=prestack

    return hblevel.shape[1], finalstack


def haar_transform(data, row_tree, norm="L2"):
    """
    Computes the Haar transform of data with respect to row_tree. (nothing
    fancy here, can go faster).
    """
    basis = compute_haar_anytree(row_tree, False, norm)
    return basis.T.dot(data)


def inverse_haar_transform(coefs, row_tree, norm="L2"):
    """
    Computes the inverse Haar transform of coefficients with respect to
    row_tree. Again nothing fancy here.
    """
    basis = compute_haar_anytree(row_tree)
    if norm == "L1":
        norm_vec = np.sum(np.abs(basis), axis=0)
        for col in range(basis.shape[1]):
            basis[:, col] /= norm_vec[col]
    return basis.dot(coefs)


def level_correspondence(row_tree):
    """
    Returns a vector of the correspondence of the haar basis vectors to the
    levels of the tree.
    """
    level_counts = [[x.size for x in row_tree.dfs_level(i)] for i in
                    range(1, row_tree.tree_depth + 1)]
    marks = [0] + [row_tree.size - sum([y - 1 for y in x]) for x in level_counts]
    z = np.zeros(row_tree.size, int)
    for (idx, t) in enumerate(marks):
        if idx == len(marks) - 1:
            z[t:] = idx + 1
        else:
            z[t:marks[idx + 1]] = idx + 1
    return z


def bihaar_transform(data, row_tree,row_root, col_tree, col_root,folder_sizes=False):
    """
    Computes the bi-Haar transform into the basis induced by row_tree and
    col_tree jointly.
    """
    Nnames = []
    for node in PreOrderIter(row_tree[row_root]):
        Nnames.append(node.name)
    roothandling_row = np.amax(np.unique(np.asarray([int(x) for x in Nnames]))) + 1
    Nnames = []
    for node in PreOrderIter(col_tree[col_root]):
        Nnames.append(node.name)
    roothandling_col = np.amax(np.unique(np.asarray([int(x) for x in Nnames]))) + 1
    if folder_sizes:
        row_hb, row_parents = compute_haar_anytree(row_tree[row_root], folder_sizes)
        row_parents[row_parents == -1] = int(row_root)#roothandling_row
        col_hb, col_parents = compute_haar_anytree(col_tree[col_root], folder_sizes)
        col_parents[col_parents == -1] = int(col_root)#roothandling_col
    else:
        row_hb = compute_haar_anytree(row_tree[row_root], folder_sizes)
        col_hb = compute_haar_anytree(col_tree[col_root], folder_sizes)
    row_transform = row_hb.T.dot(data)
    coefs = col_hb.T.dot(row_transform.T)
    if folder_sizes:
        row_sizes = np.array([len(row_tree[x].leaves) for x in row_parents.astype(str)])
        col_sizes = np.array([len(col_tree[x].leaves) for x in col_parents.astype(str)])
        return coefs.T, np.outer(row_sizes,col_sizes) / (1.0 * len([len(row_tree[x].leaves) for x in row_parents.astype(str)]) * len([len(col_tree[x].leaves) for x in col_parents.astype(str)]))
    else:
        return coefs.T


def inverse_bihaar_transform(coefs, row_tree, col_tree):
    """
    Computes the inverse bi-Haar transform of coefs.
    """
    row_hb = compute_haar_anytree(row_tree)
    col_hb = compute_haar_anytree(col_tree)
    row_transform = row_hb.dot(coefs)
    matrix = col_hb.dot(row_transform.T)
    return matrix.T

def find_node_by_name(root, name):
    """Find the node by its name."""
    return findall_by_attr(root, name, name='name')[0]

def sum_leaf_weights(node):
    """Sum the weights of all leaf nodes under a given node."""
    return sum(n.weight for n in findall_by_attr(node, True, name='is_leaf'))

def weighted_tree_distance(root, nodelist,i, j):
    """Calculate the tree distance based on the weighted sum of the smallest subtree containing both i and j."""
    # Find nodes by names i and j
    #nodelist=np.asarray([int(x.name) for x in root.leaves])
    if i==j:
        return 0.0
    node_i = root.leaves[np.where(nodelist==i)[0][0]]#find_node_by_name(root, i)
    node_j = root.leaves[np.where(nodelist==j)[0][0]]

    # Find the common ancestor of both nodes
    #w = Walker()
    ancestor = util.commonancestors(node_i, node_j)[-1]

    # Sum weights in the subtree of the common ancestor
    subtree_weight_sum = sum(x.weight for x in ancestor.leaves)
    #subtree_weight_sum = sum_leaf_weights(ancestor)

    # Sum weights in the entire tree


    # Calculate the normalized weighted tree distance
    return float(subtree_weight_sum)

def count_leaves(node):
    """Count the number of leaf nodes under a given node."""
    return sum(1 for _ in findall_by_attr(node, True, name='is_leaf'))

def tree_distance(root, nodelist, i, j):
    """Calculate the tree distance based on the smallest subtree containing both i and j."""
    # Find nodes by names i and j
    if i==j:
        return 0.0

    node_i = root.leaves[np.where(nodelist==i)[0][0]]#find_node_by_name(root, i)
    node_j = root.leaves[np.where(nodelist==j)[0][0]]

    # Find the common ancestor of both nodes
    #w = Walker()
    #ancestor = w.get_common_ancestor(node_i, node_j)
    ancestor = util.commonancestors(node_i, node_j)[-1]

    # Count the number of leaves in the subtree of the common ancestor
    subtree_leaves = len(ancestor.leaves)

    # Count the total number of leaves in the tree
    #total_leaves = len(root.leaves)

    # Calculate the normalized tree distance
    return float(subtree_leaves) #/ total_leaves

def recon_2d_haar_folder_size(data,row_tree,row_root,col_tree,col_root,threshold=0.0):
    """
    Reconstruction of a function in two variables (data) by the following:
    Find the bi-Haar expansion of data in terms of the row_tree/col_tree
    basis. Then set all coefficients corresponding to folders of
    size < threshold to 0.0, and perform the inverse transform.
    """
    coefs, folder_sizes = bihaar_transform(data,row_tree,row_root,col_tree,col_root,True)
    coefs[folder_sizes < threshold] = 0.0
    return inverse_bihaar_transform(coefs,row_tree[row_root],col_tree[col_root])

def recon_2d_haar_quantile(data,row_tree,row_root,col_tree,col_root,threshold=0.0):
    """
    Reconstruction of a function in two variables (data) by the following:
    Find the bi-Haar expansion of data in terms of the row_tree/col_tree
    basis. Then set all coefficients corresponding to folders of
    size < threshold to 0.0, and perform the inverse transform.
    """
    coefs, folder_sizes = bihaar_transform(data,row_tree,row_root,col_tree,col_root,True)
    Q=np.quantile(np.reshape(folder_sizes,-1),threshold)
    coefs[folder_sizes < Q] = 0.0
    return inverse_bihaar_transform(coefs,row_tree[row_root],col_tree[col_root])