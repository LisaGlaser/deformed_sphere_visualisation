#!/usr/bin/env python

import math
import numpy as np
import scipy.sparse
import itertools
from functools import reduce

#from twosphere.twosphere import TwoSphere
from twosphere.fuzzysphere_deformed import FuzzyTwoSphere,Dirac_state
from graphmaker.spectral_invariants.invariants import dimension, volume

"""Setup GraphmakerData for the Dirac sphere of given size."""




def prepare_gmdata(spinorsize, inputbasename=None, nthreads=1,
                     pot_coupling=3, cvxsolver_args={"eps": 10**-5,
                                                     "solver": "SCS",
                                                     "max_iters": 500000},
                     **kwargs):
    """Prepare arguments for a GraphmakerData object for the standard Dirac sphere
    of given rank."""
    sphere = FuzzyTwoSphere(spinorsize,kwargs['def'])
    D = sphere.DiracFS
    gmargs = {}
    gmargs['coordinates'] = get_coordinates(sphere) #### this is where we get the coordinates for the embedding, which are taken from the code we have
    gmargs['sq_coordinates'] = get_sq_coordinates(spinorsize,kwargs['def'])
    gmargs['D'] = D
    evals = np.linalg.eigvals(gmargs['D']).real ### this is to estimate state number
    gmargs['alg_generators'], gmargs['alg_generators_com_D'] = get_generators(spinorsize,gmargs['D'],kwargs['algebra'],kwargs['def'])
    gmargs['pot_coupling'] = pot_coupling
    gmargs['cvxsolver_args'] = cvxsolver_args
    gmargs['nthreads'] = nthreads
    gmargs['period']=0
    gmargs['npoints'],gmargs['statedata']=calc_max_points(kwargs['npoints'],evals)

    return gmargs


def calc_max_points(numPoints,evals):
    statedat=estimate_state_num_bound(evals)
    maxP=statedat[0]
    if(numPoints=='naive'):
        maxP=int(len(evals)/2) ### very naive attempt, but the numbers should be ok-ish
    elif(numPoints[:5]=='fixed'):
        __, maxP=numPoints.split('_')
        maxP=int(maxP)
    return maxP,statedat



def get_coordinates(sphere):
    """Get the coordinate matrices x, y and z of S^2 as acting on the
    Dirac eigenspinors."""
    size = sphere.spinorsize
    P = sphere.P ## The coordinates are hermitian as they come from here, no need to add more 1j!
    x=P[0]
    y=P[1]
    z=P[2]

    return [x, y, z]



def get_sq_coordinates(target_dim,defo):
    """Calculate squared coordinate matrices"""
    ## this is a leftover because this step was more complicated in the truncated case
    sphere = FuzzyTwoSphere(target_dim,defo)
    bigcoordinates = get_coordinates(sphere)
    mymats = [mat@mat for mat in bigcoordinates]
    return mymats


def eigen_dimension(l):
    return 2*l


def l_max(dim):
    return math.ceil((-1+math.sqrt(1+2*dim))/2)


def split_real_imag(matrix):
    """Split matrix into selfadjoint and antiselfadjoint parts."""
    realmat = (matrix + matrix.conjugate().T)/2
    imagmat = (matrix - matrix.conjugate().T)/2
    return realmat, -1j*imagmat


def commutator_nonzero_p(mat1, mat2):
    """True if mat1 and mat2 do _not_ commute."""
    norm = scipy.sparse.linalg.norm(mat1@mat2 - mat2@mat1)
    return not np.isclose(norm, 0)


def get_generators(target_dim,D,alg,defo):
    sphere = FuzzyTwoSphere(target_dim,defo)
    if(alg=='S2'):
        return get_spherical_harmonics(target_dim, sphere, D)
    elif(alg=='Mn'):
        #print("currently no spherical harmonics here")
        return get_naive_generators(target_dim,sphere)
    else:
        print("Not a valid value, defaulting to S2")
        return get_spherical_harmonics(target_dim, sphere, D)

### naive implementation, but if it works it works, actually creates more elements than we need
def gen_coms(mat_list,size):
    combinations=np.array(mat_list)
    combinations=np.append(combinations,[x@x for x in mat_list[:-1]],axis=0)
    for d in np.arange(3,size):
        for x in itertools.combinations_with_replacement(mat_list,d):
            mat=[reduce((lambda x, y: x@y), x)]
            combinations=np.append(combinations,mat,axis=0)
    print("unreduced length {}".format(len(combinations)))
    combinations=reduce_combinations(combinations)
    print("reduced length {}".format(len(combinations)))
    return combinations

def reduce_combinations(combinations):
    redc=combinations[:5]
    lin=[m.flatten() for m in combinations]
    linc=lin[:5]
    for x,m in zip(lin[5:],combinations[5:]):
        ## solve linear equation sum of rec
        U, s, V = np.linalg.svd(np.append(linc,[x],axis=0))
        #print(s)
        e=10e-15
        if s[-1]>e:
            redc=np.append(redc,[m],axis=0)
            linc=np.append(linc,[x],axis=0)
    return redc



def get_spherical_harmonics(target_dim, sphere, D):
    """Obtain the matrices Y_{lm} as acting on the Dirac eigenspinors."""
    realsphs = np.array(sphere.getP())
    ## I know that the generators I have do not commute with the Dirac, so I can use them immediately
    ## they are however anti hermitian. I probably want hermitian coordinates, so maybe give them an 1j?
    ## if I make them hermitian things go to hell immediately. WHYYYYY?
    realsphs=gen_coms(realsphs,target_dim)
    realsphs= [scipy.sparse.coo_matrix(mat) for mat in realsphs]
    realsphs_comD=[sphere.comm_D(mat.todense()) for mat in realsphs]
    print('got {} generators'.format(len(realsphs)))
    filter=[not np.isclose(np.linalg.norm(mat),0) for mat in realsphs_comD]
    realsphs = np.array(realsphs)[filter]
    realsphs_comD=np.array(realsphs_comD)[filter]
    print('after filter got {} generators'.format(len(realsphs)))
    realsphs_comD=np.array(realsphs_comD)
    return realsphs,realsphs_comD

def get_naive_generators(target_dim,sphere):
    """Obtain a matrix basis"""
    mylmax =target_dim
    print("naive mess")
    #print(mylmax)
    ### my naive generators are too naive. Let's start with just the basics
    ### let's use commutators
    #print("da size")
    #print(scipy.sparse.coo_matrix(([.1], ([1], [2])), shape=(target_dim,target_dim)).shape[0])
    #print(acommB(scipy.sparse.coo_matrix(([.1], ([1], [2])), shape=(target_dim,target_dim))).shape[0])
    ### why am I doing only upper diagonal? That makes no sense whatsoever
    ### i could restrict to only hermitian that would maybe help?
    realsphs1 = [scipy.sparse.coo_matrix(([1.], ([i], [j])), shape=(target_dim,target_dim)) for j in np.arange(mylmax) for i in np.arange(mylmax)]
    realsphs2 = [scipy.sparse.coo_matrix(([1j], ([i], [j])), shape=(target_dim,target_dim)) for j in np.arange(mylmax) for i in np.arange(mylmax)]
    realsphs=np.append(realsphs1,realsphs2,axis=0)
    print('got {} generators'.format(len(realsphs)))
    ### if the commutator with D is 0 that means we can add infinite amounts of it
    ### but it might still change the distanc, shouldn't it?
    ### on the other hand, if they commute with D then they and D have compatible eigenstates, and for eigenstates of D the dispersion would be 0 by default because <s|D|s>^2=l^2 <s|s>=<s|D^2|s> so then we would not want [D,a]=0 states after all
    realsphs_comD=[sphere.comm_D(mat.todense()) for mat in realsphs]
    #print([ np.linalg.norm(mat) for mat in realsphs_comD])
    filter=[not np.isclose(np.linalg.norm(mat),0) for mat in realsphs_comD]
    realsphs = np.array(realsphs)[filter]
    realsphs_comD=np.array(realsphs_comD)[filter]
    print('after filter got {} generators'.format(len(realsphs)))
    #print(realsphs[1])
    #np.savetxt('test.txt',realsphs)
    return realsphs, realsphs_comD

def acommB(mat):
    return tp(mat,np.identity(mat.shape[0]))+tp(np.identity(mat.shape[0]),np.transpose(mat))

def tp(matA,matB):
    return scipy.sparse.kron(matA,matB)



### now the tools to calculate evs etc are here

def estimate_state_num_bound(evals):
    dim = estimate_dimension(evals)
    vol = volume(evals, dim=dim)
    print("Estimated dimension {} and volume {}".format(dim,vol))
    ebv = estimate_euclidean_ball_volume(dim)

    cutoff = max(evals)
    euc_disp = estimate_euclidean_dispersion(cutoff, dim)
    es_states=vol / (ebv * euc_disp**(dim/2))
    print("Estimated number of states is {}".format(es_states))
   
    print( "anadata {} {} {} {}".format(dimension(evals),vol, es_states,ebv))
    return [es_states,dim,vol,ebv,euc_disp]


def estimate_dimension(evals):
    return round(dimension(evals))


def estimate_euclidean_dispersion(cutoff, dim):
    cov_diag = math.log(cutoff)/cutoff**2
    cov_trace = dim*cov_diag
    return cov_trace


def estimate_euclidean_ball_volume(dim):
    if dim%2==0:
        k = int(dim/2)
        denom = math.factorial(k)
        numer = math.pi**k
    else:
        k = int(dim/2+1/2)
        denom = 2*math.factorial(k) * 4* math.pi**k
        numer = math.factorial(2*k+1)

    return numer / denom
