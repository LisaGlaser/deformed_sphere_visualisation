#!/usr/bin/env python

import numpy as np
from itertools import combinations
import random

### in theory, running this should tell us the embedding dimension we need.

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def check_embedding_dim(dist_mat,max=0,cutoff=100):
    if(max==0):
        max=len(dist_mat)
    ### i from 1 to no of points:
    det_list=[calc_sub_det(dist_mat,i,cutoff=cutoff) for i in range(2,max)]
    det_av=[np.mean(x) for x in det_list]
    det_max=[np.max(x) for x in det_list]
    det_min=[np.min(x) for x in det_list]

    return [det_av,det_max,det_min,det_list]

def calc_sub_det(dist_mat,i,cutoff):
    ### pick all submatrices with i points
    if cutoff:
        rc=[random_combination(np.arange(len(dist_mat)),i) for x in range(cutoff)]
    else:
        rc=combinations(np.arange(len(dist_mat)),i)
    submats=np.array([dist_mat[np.ix_(x,x)] for x in rc])
    ### calculate determinants
    dets=[np.linalg.det(x) for x in submats]

    return dets
