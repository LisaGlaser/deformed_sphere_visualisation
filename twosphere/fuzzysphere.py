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


class memoizedSPHarmCoefficients:
    @memoize
    def threej(self, coeff):
        """Return the 3j-symbol (j1 j1 j2, m1 m2 m3)"""
        j1, m1, j2, m2, j3, m3 = coeff
        myargs = tuple(int(2*x) for x in (j1, j2, j3, m1, m2, m3))
        L1, L2, L3, M1, M2, M3 = myargs
        if M1 + M2 + M3 != 0:
            return 0
        elif abs(L1 - L2) > L3:
            return 0
        elif L1 + L2 < L3:
            return 0
        else:
            return py3nj.wigner3j(*myargs)

    def TripleProduct(self, j1, m1, s1, j2, m2, s2, j3, m3, s3):
        """Calculate int_{S^2} {}_{s_1} Y_{j1 m1} ... {}_{s_3} Y_{j3 m3}."""
        return math.sqrt((2*j1+1)*(2*j2+1)*(2*j3+1) / (4*math.pi)) * self.threej((j1, m1, j2, m2, j3, m3)) * self.threej((j1, -1*s1, j2, -1*s2, j3, -1*s3))

    def SPHarmComponents(self, l, m, l2, m2, s2, l3, m3, s3):
        """Return the triple integral {}_0 Y_{lm} {}_{s_2} Y_{l2m2}
        {}_{s3} Y_{l3 m3}."""
        s1 = 0
        j1 = l
        leftjms = (l2, m2, s2)
        rightjms = (l3, -1*m3, -1*s3)  # complex conjugation
        sign = (-1)**(m3+s3)
        return self.TripleProduct(j1, m, s1, *leftjms, *rightjms) * sign

    def SPHarmCoefficients(self, l, m,
                           l2, m2, diracsign_2,
                           l3, m3, diracsign_3):
        """Coefficients of Y_{lm} in the Dirac eigenspinor basis."""
        topleftjms = (l2, m2, 1/2)
        toprightjms = (l3, m3, 1/2)
        top = self.SPHarmComponents(l, m, *topleftjms, *toprightjms) * (diracsign_2 * diracsign_3/2)  # sign from top part (gb 9.57) and normalization
        bottomleftjms = (l2, m2, -1/2)
        bottomrightjms = (l3, m3, -1/2)
        bottom = self.SPHarmComponents(l, m, *bottomleftjms, *bottomrightjms) * (1/2)  # sign from bottom part (1j*-1j) and normalization
        return top + bottom

class FuzzyTwoSphere:
    def __init__(self, spinorsize):
        self.spinorsize = spinorsize
        self.maxspin = (spinorsize-1)/2.
        self.diracsize=spinorsize*spinorsize*4
        self.P = self.getP()
        #self.J = self.getJ()
        self.DiracFS = self.get_Dirac_FS(spinorsize)
        #self.eigenspinors = self.geteigenspinors(self.DiracFS) ## this should be the coefficients for the eigenstates.

        #self.JPJ = self.getJPJ()
        #self.G = self.getGamma()

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
            L=np.diag(enA,-1)+np.diag(enA,+1)
        elif [i,j]== [1,3]:
            enA=[c*f1(k) for k in ra[:-1]]
            enAm=[c*-1*f1(k) for k in ra[:-1]]
            L=np.diag(enAm,1)+np.diag(enA,-1)
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
                D+=tp((self.gammai(0)@self.gammai(i)@self.gammai(j))@state.spinor,comm(self.Lij(i,j,self.spinorsize),state.mat))
        return D

    def comm_D(self,a):
        le=4*len(a)
        D=1j*np.zeros([le,le])
        for i in [1,2]:
            for j in np.arange(i+1,4):
                #print("g0 g{} g{} te [L ] = {}".format(i,j,tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))))
                #print(self.Lij(i,j,self.spinorsize).shape[0])
                D+=tp((self.gammai(0)@self.gammai(i)@self.gammai(j)),comm(self.Lij(i,j,self.spinorsize),a))
                #print("This is commutator ={}".format(comm(self.Lij(i,j,self.spinorsize),a)))
        #print("This is comm ={}".format(D))
        return D

    def get_Dirac_FS(self,spinorsize):
        D=(1+0.*1j)*tp(self.gammai(0),np.identity(spinorsize*spinorsize))
        #print("g0 te m = {}".format(D))
        for i in [1,2]:
            for j in np.arange(i+1,4):
                #print("g0 g{} g{} te [L ] = {}".format(i,j,tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))))
                D+=tp(self.gammai(0)@self.gammai(i)@self.gammai(j),commB(self.Lij(i,j,spinorsize)))
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
