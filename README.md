This code needs python, with cvxpy installed.

Running the command as
```
./s2_graphmaker_def.py
```
without any options given corresponds to 
```
./s2_graphmaker_def.py --dim 12 --input None --output None --time 60 --loop 0 --algebra S2--npoints spec --deformation 1. 1. 1.
```
This generates the geometry of a non-deformed sphere and then exits the code before generating states.
To generate a small fuzzy sphere you might try the command 
```
./s2_graphmaker_def.py --dim 4 --output HelloFuzzySphere --time 2000 --loop 1 --algebra S2 --deformation 1. 1. 1.
```
For more information on the different options try 
```
./s2_graphmaker_def.py --help
```
or look into the code itself.

plotscripts contains a simple script to plot the 3d embeddings, and my script for fitting ellipses to the geometries.

# Resource usage indication example: S2
In this code Lambda=N*(N+1) since the new code uses matrix size N
instead of maximal eigenvalue Lambda as the determining factor for simulation size.
On an ordinary desktop computer (Intel Core i5-4590S, 8GB RAM), state
generation is very much feasible (<6h / state) at Lambda = 16,
i.e. with matrices of size 612 and an operator space PAP consisting of
the spherical harmonics up to l = 33.

Distance calculation with the 'SCS' algorithm uses significant amounts
of memory, however: a single thread in dimension 312 (Lambda = 12)
uses around 7 GB, which limits feasibility of higher-precision graph
generation on typical desktop systems.

