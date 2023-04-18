#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import h5py
import tables
from collections import deque
%matplotlib widget

from matplotlib.animation import FuncAnimation, PillowWriter

import pandas as pd
sns.set_style("whitegrid")

#os.chdir('/TheDirectory/Where/Your/Data/Are')
os.chdir('/home/glaser/Work/Projects/Visualisation/deformed_sphere_visualisation/')

files=glob.glob('*.hdf5')
files=np.array(files)
files.sort()
data=[h5py.File(fa, "r") for fa in files]

## loading data I might want (not all of it used in this version of the plotscript)
Nval=np.array([dada.attrs['spinorsize'] for dada in data])
emb=[np.array(d['embedding']['embedding_coords']) for d in data]
rvals=[np.array([np.linalg.norm(x) for x in d]) for d in emb]
av_r=[np.average(r) for r in rvals]
var_r=[np.var(r) for r in rvals]
dim=[d['states'].attrs['dim est'] for d in data]

disps=[np.array(fa['states']['dispersions']) for fa in data]
states=[np.array(dada['states']['vectors']) for dada in data]
alg=[fifi.attrs['algebra'] for fifi in data]
pot=[fifi.attrs['pot_coupling'] for fifi in data]
defo=[fifi.attrs['def'] for fifi in data]


coordinates=[fifi['ini_data']['coordinates'] for fifi in data]
Xmat=[c[0] for c in coordinates]
Ymat=[c[1] for c in coordinates]
Zmat=[c[2] for c in coordinates]
dists=[np.array(da['states']['distances']) for da in data]
emb_dists=[[[np.sqrt(sum((e-f)**2)) for f in em] for e in em] for em in emb]
dist_diff=[[aes-bs for aes,bs in zip(a,b)] for a,b in zip(dists,emb_dists)]
av_dist2_diff=[[np.average(xe**2) for xe in xa] for xa in dist_diff]

## make a data frame
df = pd.DataFrame(data={'Nval':Nval, 'alg': alg, 'pot':pot, 'dim':dim, 'defo':defo,'rvals':rvals, 'av_r':av_r,'var_r':var_r,'emb':emb, 'states':states,'disps':disps, 'X':Xmat, 'Y':Ymat, 'Z':Zmat, 'dists':dists, 'emb_dists':emb_dists, 'dist_diff':dist_diff, 'av_dist_diff':av_dist2_diff})
df.sort_values(by=['Nval','alg','pot'], inplace=True)

#%%
""" Here is a 3d plot"""
my_cmap = sns.light_palette("Navy", as_cmap=True)
col1=my_cmap(200)

for index,da in df.iterrows():
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(da.emb[:,0],da.emb[:,1],zs=da.emb[:,2],color=col1) 
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect('equal','box')
    plt.tight_layout()
    plotname="My_plot.pdf"
    plt.savefig(plotname)
    plt.show()

### just for fun 
### let's plot the points in color depending on their embedding quality.
my_cmap = sns.dark_palette("#79C", as_cmap=True)

def cn(c0,cc):
    return (c0-min(cc))/(max(cc)-min(cc))
from mpl_toolkits.axes_grid1 import make_axes_locatable

for index,da in df.iterrows():
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ## calculate correlation coefficients as proxy for embedding quality
    cc=deque()
    for xo,po,poe in zip(np.arange(len(da.dists)),da.dists,da.emb_dists):
        cc.append(np.corrcoef(po,poe)[0,1])
    cc=np.array(cc)
    temp=ax.scatter(da.emb[:,0],da.emb[:,1],zs=da.emb[:,2],c=cc,cmap=my_cmap)
    xp,yp,zp= projection_points(da.emb[:,0],da.emb[:,1],da.emb[:,2]) 
    for x1,y1,z1,xp1,yp1,zp1,coco in zip(da.emb[:,0],da.emb[:,1],da.emb[:,2],xp,yp,zp,cc):
       ax.plot([x1,xp1],[y1,yp1],[z1,zp1],color=temp.cmap(cn(coco,cc)))
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect('equal','box')
    fig.colorbar(temp,orientation="horizontal",fraction=0.046, pad=0.04,label="Correlation Coefficient")
    plotname="Correlation_colored_points.pdf"
    plt.savefig(plotname)
    plt.show()
#%%

### close the data
for d in data:
    d.close()

# %%
