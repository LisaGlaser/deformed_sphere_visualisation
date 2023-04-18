#!/usr/bin/env python

import argparse
import math
import multiprocessing
import copy
import time
import subprocess
import h5py
import numpy as np

from s2_init_def import prepare_gmdata,get_generators
from graphmaker.graphmaker import (increment_graph, as_distancearray,
                                   pprint,as_data_arrays)
from graphmaker.smacof.smacof import (SMACOF, run_embedding,
                                      WeightedDistanceGraph)
from graphmaker.smacof.smacof import writeout as smacof_write_bin
from graphmaker.graphmaker import GraphmakerData
from graphmaker.localized_states import State


"""Construct and embed metric graphs associated to (PC(S^2)P,
PL^2(S^2,S), PD_{S^2}."""


def make_embedding(gmdata, outputbasename):
    """Run SMACOF to embed the graph in R^3."""
    darray = as_distancearray(gmdata)
    graph = WeightedDistanceGraph(darray,
                                  weight_function=lambda source, target, dist: math.exp(-dist))
    embedder = SMACOF(graph, dimension=3)

    run_embedding(embedder, num_steps=1000)
    smacof_write_bin(outputbasename, embedder)

    pprint("Writing output to {}.".format(
        outputbasename + " smacof.csv"))
    return


def parse_arguments():
    # Define command-line options
    description = "Calculate Connes distance for S2"
    parser = argparse.ArgumentParser(description=description)

    dim_help = "Dimension"
    input_help = "File with initial graph."
    output_help = "Basename for output files"
    parallel_help = "Number of parallel calculations"
    time_help = "Max calculation duration"
    mindistance_help = "Minimal distance between states"
    potential_help = "Strengths of repulsion between states"
    looping_help = "Run the graphmaker now!"
    hook_help = "Run $cmd upon adding data"
    algebra_help = "S2 uses truncated spherical harmonics, Mn uses arbitrary complete matrix algebra (slower!)"
    npoints_help= "How is the number of points calculated, or are they fixed (options spec fix_NUM naive )"
    def_help= "For the deformed sphere, how are the directions changed"

    def_threadnum = multiprocessing.cpu_count()
    parser.add_argument('--dim', metavar='dim', type=int,
                        default=12, help=dim_help)
    parser.add_argument('--input', metavar='input', type=str,
                        default=None, help=input_help)
    parser.add_argument('--output', metavar='output', type=str,
                        default=None, help=output_help)
    parser.add_argument('--threads', metavar='threads', type=int,
                        default=def_threadnum, help=parallel_help)
    parser.add_argument('--time', metavar='time', type=int,
                        default=60, help=time_help)
    parser.add_argument('--max-dispersion', metavar='max-dispersion',
                        type=float, default=0.3,
                        help=mindistance_help)
    parser.add_argument('--potential', metavar='electrostatic potential',
                        type=float, default=10,
                        help=potential_help)
    parser.add_argument('--loop', metavar="loop", type=int,
                        default=0, help=looping_help)
    parser.add_argument('--hook', metavar="hook", type=str,
                        default=None, help=hook_help)
    parser.add_argument('--algebra', metavar="algebra", type=str,
                        default='S2', help=algebra_help)
    parser.add_argument('--npoints', metavar="npoints", type=str,
                        default='spec', help=npoints_help)
    parser.add_argument('--deformation', metavar="deformation of sphere", type=float, nargs='+', default=[1.,1.,1.], help=def_help)

    args = parser.parse_args()
    options = {}
    options['spinorsize'] = args.dim
    options['inputbasename'] = args.input
    options['outputbasename'] = args.output
    options['nthreads'] = args.threads
    options['duration'] = args.time
    options['max_dispersion'] = args.max_dispersion
    options['pot_coupling'] = args.potential
    options['loop'] = args.loop
    options['hook'] = args.hook
    options['algebra'] = args.algebra
    options['npoints'] = args.npoints
    options['def'] = args.deformation

    return options


def hdf5_options(options,gmda):
    ### create a .hdf file

    hf = h5py.File("{}.hdf5".format(options['outputbasename']), 'w')
    #### Which data do I want in the file:
    ##  input data: all data used in the input so , all data for cvx_solver so basically cvxsolver_args

    #g1=hf.create_group('states')
    for key in options:
        if options[key]!=None:
            hf.attrs.create(key,data=options[key])
    for key in gmda['cvxsolver_args']:
        hf.attrs.create("cvx_op_{}".format(key),data=gmda['cvxsolver_args'][key])

    g1=hf.create_group('ini_data')

    for key in ['coordinates','sq_coordinates','D']:
        ### TODO I think this only saves the first line of the data, need to figure out how to fix that.
        g1.create_dataset(key,data=gmda[key])
        #print(key)
        #print(gmda[key])
    g1.create_dataset('alg_generators',data=[a.toarray() for a in gmda['alg_generators']])
    ## missing cvx_solver data
    hf.close()


def run_hook(hook, outputbasename):
    """Run the program specified by hook with arg outputbasename."""
    if hook:
        subprocess.run([hook, outputbasename])


def save_as_hdf5(gmdata,outfile):
    ### open .hdf file
    hf = h5py.File("{}.hdf5".format(outfile), 'a')
    g1=hf.create_group('states')
    for key,da in zip(['max_states_naive','dim est','vol est','est euclid ball volume', 'euclid dispersion' ],gmdata.statedata):
        g1.attrs.create(key,data=da)
    g1.attrs.create('max_states',gmdata.maxstates)
    #### get all the states data
    arraydat=as_data_arrays(gmdata)
    for key in arraydat:
        print(key)
        g1.create_dataset(key,data=arraydat[key])

    hf.close()


def loop_expand_big_graph(gmdataDb, outputbasename, duration, hook):
    start_time = time.time()

    if outputbasename:
        outputDb = outputbasename

    def elapsed_time():
        return int(time.time() - start_time)

    if len(gmdataDb.states) > gmdataDb.maxstates:
        pprint("The maximal number of states is {} and we already have {}".format(gmdataDb.maxstates,len(gmdataDb.states)))

    ## added a +1 so that the added state does not break over the maximal number
    while elapsed_time() < duration and len(gmdataDb.states)+1 < gmdataDb.maxstates:
        pprint("Have {} of {} expected states. Stepping once. {} seconds left.".format(
            len(gmdataDb.states),gmdataDb.maxstates,
            duration - elapsed_time()))
        dispersion=increment_graph(gmdataDb, steps=1)
        gmdataDb.set_maxstates(dispersion)
        #if outputbasename:
        #    writeout(gmdataDb, outputDb)

        run_hook(hook, outputbasename)

    if outputbasename:
        make_embedding(gmdataDb, outputDb)
        run_hook(hook, outputbasename)

    save_as_hdf5(gmdataDb,outputDb)

    return


if __name__ == "__main__":
    options = parse_arguments()
    pprint("Constructing gmdata...")
    gmargs_Db = prepare_gmdata(**options)
    gmdata_Db = GraphmakerData(**gmargs_Db)

    pprint("Done initializing.")

    if options['inputbasename'] is not None:
        pprint("Reading from {}...".format(options['inputbasename']))
        try:
            file=h5py.File(options['inputbasename'], "r")
            rawstates=[list(s) for s in file['states'].values()]
            print(len(rawstates[0]))
            states=[State(np.block([vec.real, vec.imag]), potential_at_creation=pot,dispersion=disp) for disp,dist,pot,vec in zip(*rawstates)]
            distances={}
            rawdistances =np.array(file['states']['distances'])
            for n, state1 in enumerate(states):
                for m, state2 in enumerate(states):
                    distances[state1, state2] = rawdistances[n, m]
            gmdata_Db.states = states
            gmdata_Db.distances = distances
            ini_disp=[disp for disp,dist,pot,vec in zip(*rawstates)][-1]
            gmdata_Db.set_maxstates(ini_disp)

        except FileNotFoundError:
            print("Input file does not exist, starting from scratch.")


    outputbasename = options['outputbasename']
    hook = options['hook']

    hdf5_options(options,gmargs_Db)

    def loop_once(duration=options['duration']):
        loop_expand_big_graph(gmdata_Db,
                              outputbasename, duration, hook)

    if options['loop']:
        loop_once(options['duration'])
