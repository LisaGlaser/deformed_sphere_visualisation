#!/usr/bin/env python

import numpy as np
from .connes_distance import DistanceParameters, find_distance
from .localized_states import StateFinderParameters, State, find_new_state
from .density_estimator.estimator import estimate_density_radius
import math
import itertools
from multiprocessing import Pool
import functools
import time
import operator


"""Friendly helper library around connes_distance and localized_states.

increment_graph() creates a distance graph, write_bin/npz() and read_bin/npz()
write/read a graph from disk."""


def round_to_zero(myarray, tolerance):
    """Round array to zero within tolerance. Is useful for sparsity."""

    smallreals = np.abs(myarray.real < tolerance)
    smallimags = np.abs(myarray.imag < tolerance)
    result = myarray
    result.real[smallreals] = 0
    result.imag[smallimags] = 0
    return result


class GraphmakerData(DistanceParameters,
                     StateFinderParameters):
    """Contains both the distance graph and the parameters required to
    grow it."""

    def __init__(self, nthreads=1, **kwargs):
        DistanceParameters.__init__(self, **kwargs)
        StateFinderParameters.__init__(self, **kwargs)
        self.states = []
        self.distances = {}
        self.period=kwargs['period']
        self.nthreads = nthreads
        self.statedata= kwargs['statedata']
        self.maxstates =kwargs['npoints']

    def set_maxstates(self,disp):
        dim = self.statedata[1]
        vol = self.statedata[2]
        ebv = self.statedata[3]
        es_states=vol / (ebv * disp**(dim/2))
        #pprint( "calculated using dim={} vol={} ebv={}".format(dim,vol,ebv))
        #pprint("Estimated number of state1s is {} vs naive {}".format(es_states,self.statedata[0]))
        self.maxstates=es_states
        return es_states


def append_state_inplace(gmdata):
    state = find_new_state(gmdata)
    gmdata.states.append(state)
    gmdata.reset_x0()
    fb_string = "State found, dispersion {:5.6f}, potential {:6.3f}."
    pprint(fb_string.format(state.dispersion,
                            state.potential_at_creation))
    return state.dispersion


def star_find_distance(gmdata, combination):
    return find_distance(gmdata, *combination)


def find_distances_inplace(gmdata, combinations):
    """Calculate distances from list of pairs."""
    fb_string = """getdistances got {} combinations, calculating..."""
    pprint(fb_string.format(len(combinations)))

    distfinder = functools.partial(star_find_distance, gmdata)

    with Pool(gmdata.nthreads) as p:
        result = p.imap(distfinder, combinations)
        for pair, distance in zip(combinations, result):
            gmdata.distances.update({pair: distance})
            gmdata.distances.update({tuple(reversed(pair)): distance})
            pprint("distance is {}".format(gmdata.distances[tuple(reversed(pair))]))
        pprint("...done.")


def finish_distance_graph_inplace(gmdata, max_dist_calculations=None):
    """Calculate missing distances, append to gmdata.distances."""
    states, dists = gmdata.states, gmdata.distances
    trivial_distances = {(state, state): 0 for state in states}
    dists.update(trivial_distances)
    combinations = itertools.combinations(states, r=2)
    existing_combinations = [pair for pair in dists.keys() if dists[pair]]
    missing_combinations = [pair for pair in combinations if pair not
                            in existing_combinations]
    if max_dist_calculations:
        missing_combinations = missing_combinations[0:max_dist_calculations]
        #print("Missing combinations")
        #print(missing_combinations)
    find_distances_inplace(gmdata, missing_combinations)


def pprint(string):
    prefix = time.strftime("%x %X") + ": "
    print(prefix + string, flush=True)


def increment_graph(gmdata, steps=1, max_dist_calculations=None):
    """Increment the graph in gmdata."""

    for step in range(steps):
        pprint("Adding state...")
        dispersion=append_state_inplace(gmdata)
    pprint("Updating distances...")
    finish_distance_graph_inplace(gmdata, max_dist_calculations)
    pprint("Succeeded, appending: we have {} states.".format(
        len(gmdata.states)))
    return dispersion


def as_distancearray(gmdata):
    """Return distances as array, in order distancearray[n, m] =
    distances[gmdata.states[n], gmdata.states[m]]."""

    data = as_data_arrays(gmdata)
    return data['distances']


def as_data_arrays(gmdata):
    """Export states, distances to numpy arrays, such that
    data['distances'][n, m] = distances[data['states'][n],
    data['states'][m]]."""

    states = gmdata.states
    distances = gmdata.distances

    dim = states[0].vector.shape[0]

    vectshape = (len(states), dim)
    vectarray = np.full(vectshape, np.nan, dtype=np.complex_)
    potarray = np.full(len(states), np.nan, dtype=np.float_)
    disparray = np.full(len(states), np.nan, dtype=np.float_)

    distshape = (len(states), len(states))
    distancearray = np.full(distshape, np.nan, dtype=np.float_)

    for n, state1 in enumerate(states):
        vectarray[n] = state1.vector
        disparray[n] = state1.dispersion
        potarray[n] = state1.potential_at_creation
        for m, state2 in enumerate(states):
            if (state1, state2) in distances.keys():
                distancearray[n, m] = distances[state1, state2]
    data = {}
    if states:
        data['vectors'] = vectarray
        data['dispersions'] = disparray
        data['potentials'] = potarray
    if distances:
        data['distances'] = distancearray

    return data


def from_data_arrays(data):
    """Transform data dictionary into states, distances objects."""
    states, distances = [], {}
    try:
        vectarray = data['vectors']
        disparray = data['dispersions']
        potarray = data['potentials']
        distancearray = data['distances']

        statedict = {}
        distances = {}
        iterator = enumerate(zip(vectarray, potarray, disparray))
        for n, (vect, pot, disp) in iterator:
            splitvect = np.block([vect.real, vect.imag])
            newstate = State(splitvect,
                             potential_at_creation=pot,
                             dispersion=disp)
            statedict[n] = newstate
        states = list(statedict.values())
        for n, state1 in statedict.items():
            for m, state2 in statedict.items():
                try:
                    distances[state1, state2] = distancearray[n, m]
                except IndexError:
                    print("Distance {}, {}: not yet calculated.".format(n, m))
    except KeyError:
        print("Missing part of the data.")

    return states, distances


def read_npz(basename):
    """Read from .npz output."""
    data = np.load(basename + ".npz")
    states, distances = from_data_arrays(data)
    return states, distances


def read_bin(basename):
    """Read from (older) numpy binary output."""
    data = {}
    cpkeys = {'vectors': "_states"}
    flkeys = {'dispersions': '_dispersions',
              'potentials': '_potentials', 'distances': ''}
    for key, val in cpkeys.items():
        data[key] = np.fromfile(basename + val, dtype=np.complex_)
    for key, val in flkeys.items():
        data[key] = np.fromfile(basename + val, dtype=np.float_)

    nstates = len(data['potentials'])
    ndistances = int(math.sqrt(data['distances'].size))
    data['distances'] = data['distances'].reshape((ndistances, ndistances))
    data['vectors'] = data['vectors'].reshape((-1, nstates)).T

    states, distances = from_data_arrays(data)
    return states, distances
