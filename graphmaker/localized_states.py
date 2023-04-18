#!/usr/bin/python

import numpy as np
import scipy.optimize
import scipy.linalg
import functools
import math


"""Generate localized states of an operator system spectral triple
(PAP, PH, PD)"""


def complexform(vect):
    # emulate complex vector space (of half the dimension) for numpy.optimize
    myl = int(vect.shape[0]/2)
    return vect[:myl] + 1j*vect[myl:]


def apply_state(vect, mat):
    # vect, mat -> <v|mat|v>
    return np.inner(vect, np.dot(mat, vect.conjugate()))


def stateform(vect):
    # vect -> (mat -> <v|mat|v>)
    v = complexform(vect)
    return functools.partial(apply_state, v)


class State:
    """Given v, store the state <v, - v>. Can be hashed."""

    def __init__(self, vect, potential_at_creation=None, dispersion=None, **kwargs):
        self.function = stateform(vect)
        self.vector = complexform(vect)
        self.raw=vect
        self.potential_at_creation = potential_at_creation
        self.dispersion = dispersion

    def __key(self):
        return tuple(self.vector.tolist())

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.__key() == other.__key()

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    def __str__(self):
        return "{}".format(self.raw)


class DispersionParameters:
    def __init__(self, coordinates, sq_coordinates, **kwargs):
        self.coordinates = coordinates
        self.sq_coordinates = sq_coordinates


def dispersion(params, state):
        "Calculate the dispersion, i.e. the variance of position."""
        ## now I just need to figure out what coordinates are ;)
        esqs = (state(sq_coordinate) for sq_coordinate in
                params.sq_coordinates)
        sqes = (state(coordinate)**2 for coordinate in
                params.coordinates)
        #print(params.sq_coordinates[0])
        #print(len(params.sq_coordinates[0]))
        #print("test")
        #print(sum(esqs))
        #print(sum(sqes))
        d = abs(sum(esqs) - sum(sqes))
        return d


class PotentialParameters(DispersionParameters):
    def __init__(self, coordinates, sq_coordinates, states=[],
                 min_dispersion=0, pot_coupling=100, **kwargs):
        DispersionParameters.__init__(self, coordinates, sq_coordinates)
        self.states = states
        self.min_dispersion = min_dispersion
        self.pot_coupling = pot_coupling


def potential(params, vect):
    period=params.period
    state = stateform(normalize(vect)) ### this is a function that takes a mat.
    # modeled as a pair of opposite charges of separation disp:
    # electrostatic attraction

    attraction = -1/(dispersion(params, state) + params.min_dispersion) ## so he added a minimal dispersion to avoid 0, well...
    #for st in params.states:
    #    print(st)
    #    print(coord_distance(params.coordinates, state, st))
    cod=[coord_distance(params.coordinates, state, alt_state,period) if coord_distance(params.coordinates, state, alt_state)>10**(-15) else 10**(-15) for alt_state in params.states]
    #print(cod)
    repulsion = (1/c for c in cod )
    return attraction + params.pot_coupling*sum(repulsion)


def normalize(vect):
    """Normalize non-zero vector."""
    norm = np.linalg.norm(vect)
    return vect / norm


def coord_distance(coordinates, state1, state2,period=0):
    """Return euclidean distance between states."""

    if period==0:
        d = ((state1(coordinate).real - state2(coordinate).real) for coordinate in coordinates)
        distsqs= [di**2 for di in d]
    else:
        d = (np.abs(state1(coordinate).real%period - state2(coordinate).real%period) for coordinate in coordinates)
        distsqs= [di**2 if np.abs(di)<0.5 else (1-np.abs(di))**2 for di in d]
    return math.sqrt(sum(distsqs))



class StateFinderParameters(PotentialParameters):
    def __init__(self, **kwargs):
        PotentialParameters.__init__(self, **kwargs)
        self.size = 2*self.coordinates[0].shape[0] ## twice the size of the coordinates
        constraint_lb = 0.8
        constraint_ub = 1.2
        self.constraint = scipy.optimize.NonlinearConstraint(
            fun=self.constraint_fun,
            lb=constraint_lb, ub=constraint_ub)
        self.reset_x0()

    def reset_x0(self):
        init_vect = np.random.rand(self.size)
        self.init_vect = init_vect / np.linalg.norm(init_vect)

    def constraint_fun(self, vect):
        return np.linalg.norm(vect)


def find_new_state(params):
    """Find a new state, as local minimum of the potential."""
    ### editing this really messes with the dispersion! it's epsilon. if i change that it goes bad
    opts = {'maxiter': 5000000, 'ftol':0.01} #,'eps':10}
    mypotential = functools.partial(potential, params)
    bounds = [(-1, 1) for x in range(params.size)]
    ### this is where something goes wrong. The dirac has matrix size of the square of the spinorsize
    ### I need to really get my at together on all of those things.
    success=False
    failcount=0
    ### I think the best explanation is that a random state
    ### is not a good approximation for my mess.
    ### lets get a different guess and hope
    while not success and failcount<10:
        params.reset_x0()
        result = scipy.optimize.minimize(fun=mypotential,
        x0=params.init_vect,
        constraints=params.constraint,
        options=opts, bounds=bounds)
        success=result.success
        if not success:
            failcount+=1
            print("minimze failed the {}th time with:".format(failcount))
            ### one thing that fails is if the potential is too large
            print(params.constraint.fun(result.x))
            print(result.x-params.init_vect)
            print(result.x)
            print(mypotential(result.x))
            print(result.message)

    vect = normalize(result.x)
    state = State(vect, potential_at_creation=mypotential(vect))
    state.dispersion = dispersion(params, state)
    return state
