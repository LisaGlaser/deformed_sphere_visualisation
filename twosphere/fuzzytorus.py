#!/usr/bin/env python
import numpy as np
import math
from operator import itemgetter
import scipy
import scipy.special
import py3nj
from functools import partial



class Dirac_state:
    def __init__(self, cliff,mat):
        self.spinor=cliff
        self.mat=mat
        self.spinorsize=mat.shape[0]

    def __add__(self,other):
        return Dirac_state(self.spinor+other.spinor,self.mat+other.mat)

    def __mul__(self, other):
        return Dirac_state(self.spinor@ohter.spinor,self.mat@other.mat)

    def apply_algebra_element(self,a):
        return Dirac_state(self.spinor,self.mat@a)

    def apply_gamma(self,g):
        return Dirac_state(self.spinor@g,self.mat)

    def __str__(self):
        return "v={} \n mat={} \n".format(self.spinor,self.mat)


###%%%
#
# def matFromCoeffs(coeffs, eigenspinors):
#     """Create matrix, given a function <lp,mp,sp|l,m,s> -> num."""
#
#     spinorsize = len(eigenspinors)
#     vectorcoeffs = np.vectorize(lambda i, j: coeffs(*eigenspinors[i], *eigenspinors[j]))
#     mat = np.fromfunction(vectorcoeffs, (spinorsize, spinorsize), dtype=int)
#     return mat

class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class FuzzyTorus:
    def __init__(self, spinorsize):
        self.spinorsize = spinorsize
        self.maxspin = (spinorsize-1)/2.
        self.diracsize=spinorsize*spinorsize*4
        self.Q = np.exp(2*np.pi/spinorsize*1j)
        self.P = self.getP()
        self.DiracFS = self.get_Dirac_FS()

    def getP(self):
        #pref=1j/np.sqrt(l*(l+1))
        #pref=1j
        #mat_x = pref*self.clock() #matFromCoeffs(gen_bcoeff, self.eigenspinors)
        #mat_y = pref*self.shift() #matFromCoeffs(gen_acoeff, self.eigenspinors)
        mat_x = self.getX() #matFromCoeffs(gen_bcoeff, self.eigenspinors)
        mat_y = self.getY() #matFromCoeffs(gen_acoeff, self.eigenspinors)
        return [mat_x,mat_y]

    def getX(self):
        ### this is more or less a guess, inspired by I want X, Y to be coordinates so (anti?) self adjoint
        ### and I expect their eigenvalues should be 0 to 1 for a torus with that range (but I have -1 to 1 now hmm)
        return 1/4*(self.clock()+self.clock().transpose().conjugate())

    def getY(self):
        ### this is more or less a guess, inspired by I want X, Y to be coordinates so (anti?) self adjoint
        ### and I expect their eigenvalues should be 0 to 1 for a torus with that range (but I have -1 to 1 now hmm)
        return 1/4*(self.shift()+self.shift().transpose().conjugate())

    def Kij(self,ij):
        ''' I did check them out, they work just fine'''
        Y=self.shift()
        X=self.clock()
        Xs=X.conj().transpose()
        Ys=Y.conj().transpose()
        if ij==[1] or ij==[2,3,4]:
            K=-1/4.*(X+Xs)
        elif ij==[2] or ij==[1,3,4]:
            K=-1j/4.*(Xs-X)
            if ij==[1,3,4]:
                K=-K
        elif ij==[3] or ij==[1,2,4]:
            K=1/4*(Y+Ys)
            if ij==[1,2,4]:
                K=-K
        elif ij==[4] or ij==[1,2,3]:
            K=1j/4*(Ys-Y)
        else:
            print("Your Indices are messed up. Try again")
            return "nope"
        return K



    def geteigenspinors(self,D):
        eval,evec=np.linalg.eigh(D)
        return evec


    def gammai(self,i):
        ''' I did check them out, they work ust fine'''
        if(i==1):
            gamma=1j*np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        elif(i==4):
            gamma=np.array([0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        elif(i==3):
            gamma=1j*np.array([0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0])
        else :
            gamma=np.array([0, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0])

        gamma=np.reshape(gamma,[4,4])
        return gamma

    def apply_D_FS(self,state):
        D=(0.*1j)*tp(self.gammai(0)@state.spinor,state.mat)
        f1=1/(self.Q**(1/4)-self.Q**(-1/4))
        f2=1/(self.Q**(1/4)+self.Q**(-1/4))
        for i in np.arange(1,5):
            D+=f1*tp(self.gammai(i)@state.spinor,comm(self.Kij([i]),state.mat))
        for i in np.arange(1,5):
            for j in np.arange(i+1,5):
                for k in np.arange(j+1,5):
                    D+=f2*tp(self.gammai(i)@self.gammai(j)@self.gammai(k)@state.spinor,comm(self.Kij([i,j,k]),state.mat))
        return D

    def comm_D(self,a):
        matsize=len(a)
        spinorsize=4*matsize
        if(spinorsize!= self.spinorsize):
            print("Are you sure the sizes match?")
        D=1j*np.zeros([spinorsize,spinorsize])
        f1=1/(self.Q**(1/4)-self.Q**(-1/4))
        f2=1/(self.Q**(1/4)+self.Q**(-1/4))
        for i in np.arange(1,5):
            D+=f1*tp(self.gammai(i),comm(self.Kij([i]),a))

        for i in np.arange(1,5):
            for j in np.arange(i+1,5):
                for k in np.arange(j+1,5):
                    D+=f2*tp(self.gammai(i)@self.gammai(j)@self.gammai(k),comm(self.Kij([i,j,k]),a))
        return D

    def get_Dirac_FS(self):
        D=(0.*1j)*tp(self.gammai(0),np.identity(self.spinorsize*self.spinorsize))
        f1=1/(self.Q**(1/4)-self.Q**(-1/4))
        f2=1/(self.Q**(1/4)+self.Q**(-1/4))
        for i in np.arange(1,5):
            D+=f1*tp(self.gammai(i),commB(self.Kij([i])))

        for i in np.arange(1,5):
            for j in np.arange(i+1,5):
                for k in np.arange(j+1,5):
                    D+=f2*tp(self.gammai(i)@self.gammai(j)@self.gammai(k),commB(self.Kij([i,j,k])))
        return D

    def clock(self):
        C=np.diag([self.Q**n for n in np.arange(self.spinorsize)])
        return C

    def shift(self):
        S=np.eye(self.spinorsize,k=-1)+np.eye(self.spinorsize,k=self.spinorsize-1)
        return S


def is_hermitian(mat):
    return not np.any(np.around(mat-np.transpose(np.conjugate(mat)),12))


def is_anti_hermitian(mat):
    return not np.any(np.around(mat+np.transpose(np.conjugate(mat)),12))


def doubleMatrix(mat):
    """Lift matrix to H oplus H."""

    dims = mat.shape
    myZero = np.zeros(dims)
    ans = np.block([[mat, myZero], [myZero, mat]])
    return ans

def commB(mat):
    return tp(mat,np.identity(len(mat)))-tp(np.identity(len(mat)),np.transpose(mat))

def acommB(mat):
    return tp(mat,np.identity(len(mat)))+tp(np.identity(len(mat)),np.transpose(mat))


def comm(mat1, mat2):
    return mat1@mat2 - mat2@mat1

def tp(A,B):
    return np.kron(A,B)

def relTrace(mat, spinorsize):
    """Return the relative trace [[A,B],[C,D]]->A+D."""

    top = mat[:spinorsize, :spinorsize]
    bottom = mat[spinorsize:, spinorsize:]
    return 2*(top+bottom)
