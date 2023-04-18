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



class FuzzyTwoSphere:
    def __init__(self, spinorsize,de):
        self.spinorsize = spinorsize
        self.maxspin = (spinorsize-1)/2.
        self.de=de
        print(de)
        self.diracsize=spinorsize*spinorsize*4
        self.P = self.getP()
        #self.J = self.getJ()
        self.DiracFS = self.get_Dirac_FS(spinorsize)
        #self.eigenspinors = self.geteigenspinors(self.DiracFS) ## this should be the coefficients for the eigenstates.

        #self.JPJ = self.getJPJ()
        #self.G = self.getGamma()

    def deformer(self,i,j):
        if [i,j]== [2,3]:
            return self.de[0]
        elif [i,j]== [1,3]:
            return self.de[1]
        elif [i,j]== [1,2]:
            return self.de[2]

    def getP(self):
        l=(self.spinorsize-1)/2 ### using Diracsize here might work, but is probably physically not so sensible. still best idea I have
        pref=1j/np.sqrt(l*(l+1))
        # mat_x = pref*self.Lij(2,3,self.diracsize) #matFromCoeffs(gen_bcoeff, self.eigenspinors)
        # mat_y = pref*self.Lij(1,3,self.diracsize) #matFromCoeffs(gen_acoeff, self.eigenspinors)
        # mat_z = pref*self.Lij(1,2,self.diracsize) #matFromCoeffs(gen_acoeff, self.eigenspinors)
        # ones = np.eye(self.diracsize)
        mat_x = pref*self.Lij(2,3,self.spinorsize) #matFromCoeffs(gen_bcoeff, self.eigenspinors)
        mat_y = pref*self.Lij(1,3,self.spinorsize) #matFromCoeffs(gen_acoeff, self.eigenspinors)
        mat_z = pref*self.Lij(1,2,self.spinorsize) #matFromCoeffs(gen_acoeff, self.eigenspinors)
        ones = np.eye(self.spinorsize)
        P = [
            [(ones-mat_z)/2, (mat_x-1j*mat_y)/2],
            [(mat_x+1j*mat_y)/2, (ones-mat_z)/2]
            ]
        return [mat_x,mat_y,mat_z]

    def Lij(self,i,j,spinorsize):
        ''' I did check them out, they work just fine'''
        ### need to check probably I don't want spinor size but matrix size, so a smaller number.
        ### really need to write all of those and their relation down!
        #angular momentum j
        #spinor size 2j+1
        sp=spinorsize
        jp=0.5*(spinorsize-1)
        ra=np.arange(0,sp)
        c=1 ## sign
        ### getting the sign
        if i>j:
            k=i
            i=j
            j=k
            c=-1


        def f1(k):
            return np.sqrt(((2*jp+2))*(k+1)- (k+2)*(k+1) )/2.
        L=np.zeros((sp,sp))

        if [i,j]== [2,3]:
            enA=[c*1j*f1(k) for k in ra[:-1]]
            L=(np.diag(enA,-1)+np.diag(enA,+1))
        elif [i,j]== [1,3]:
            enA=[c*f1(k) for k in ra[:-1]]
            enAm=[c*-1*f1(k) for k in ra[:-1]]
            L=(np.diag(enAm,1)+np.diag(enA,-1))
        elif [i,j]== [1,2]:
            enA=[c*1j*(jp-k) for k in ra]
            L=np.diag(enA)

        return L



    def geteigenspinors(self,D):
        eval,evec=np.linalg.eigh(D)
        return evec


    def gammai(self,i):
        ''' I did check them out, they work ust fine'''
        if(i==0):
            gamma=np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        elif(i==1):
            gamma=np.array([0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        elif(i==2):
            gamma=1j*np.array([0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0])
        else :
            gamma=np.array([0, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0])

        gamma=np.reshape(gamma,[4,4])
        return gamma

    def apply_D_FS(self,state):
        D=(1+0.*1j)*tp(self.gammai(0)@state.spinor,state.mat)
        #print("g0 te m = {}".format(D))
        for i in [1,2]:
            for j in np.arange(i+1,4):
                #print("g0 g{} g{} te [L ] = {}".format(i,j,tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))))
                D+=tp((self.gammai(0)@self.gammai(i)@self.gammai(j))@state.spinor,self.deformer(i,j)*comm(self.Lij(i,j,self.spinorsize),state.mat))
        return D

    def comm_D(self,a):
        le=4*len(a)
        D=1j*np.zeros([le,le])
        for i in [1,2]:
            for j in np.arange(i+1,4):
                #print("g0 g{} g{} te [L ] = {}".format(i,j,tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))))
                #print(self.Lij(i,j,self.spinorsize).shape[0])
                D+=tp((self.gammai(0)@self.gammai(i)@self.gammai(j)),comm(self.deformer(i,j)*self.Lij(i,j,self.spinorsize),a))
                #print("This is commutator ={}".format(comm(self.Lij(i,j,self.spinorsize),a)))
        #print("This is comm ={}".format(D))
        return D

    def get_Dirac_FS(self,spinorsize):
        D=(1+0.*1j)*tp(self.gammai(0),np.identity(spinorsize*spinorsize))
        #print("g0 te m = {}".format(D))
        for i in [1,2]:
            for j in np.arange(i+1,4):
                #print("g0 g{} g{} te [L ] = {}".format(i,j,tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))))
                D+=tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.deformer(i,j)*self.Lij(i,j,spinorsize)))
        return D

    # def SPHarm(self, l, m):
    #     """Return the matrix of the spherical harmonic Y_{lm} in the
    #     eigenspinor basis."""
    #
    #     memSPHarCoef = memoizedSPHarmCoefficients()
    #     SPHarmCoefficients = memSPHarCoef.SPHarmCoefficients
    #
    #     def mycoeffs(l0, m0, s0, l1, m1, s1):
    #         return SPHarmCoefficients(l, m, l0, m0, s0, l1, m1, s1)
    #     mat = matFromCoeffs(mycoeffs, self.eigenspinors)
    #
    #     return mat



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
