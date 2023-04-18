import numpy as np
import scipy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import os
import glob
import h5py
from string import Template
sns.set_style("whitegrid")
%matplotlib widget


os.chdir('/TheDirectory/Where/Your/Data/Are')
## carful, this picks up all data
files=glob.glob('*.hdf5')
files=np.array(files)
files.sort()
data=[h5py.File(fa, "r") for fa in files]

defo=[fifi.attrs['def'] for fifi in data]
d0=np.array([d[0] for d in defo])
npoints=np.array([len(d['states']['dispersions']) for d in data])
embedding=[ np.array(d['embedding']['embedding_coords']) for d in data]
df=pd.DataFrame({'defo':defo,'np':npoints, 'emb':embedding})


### start with a rotation matrix from a vector
def rotate(vec):
    r=np.linalg.norm(vec)
    cy=vec[2]/r
    cz=vec[0]/np.sqrt(vec[0]**2+vec[1]**2)
    sy=-np.sin(np.arccos(cy))
    sz=-np.sin(np.arccos(cz))
    maty=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    matz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return np.dot(maty,matz)

### rotates all coordinates so the longest vector points in z direction
def trans_vec(coords):
    r=np.array([np.linalg.norm(p) for p in coords])
    max_i=r.argsort()[-1]
    matrix=rotate(coords[max_i])
    return np.array([np.dot(matrix,p) for p in coords])

#%%% rotation matrix for the fit
def rot_mat(t1,t2):
    cy=np.cos(t1)
    cz=np.cos(t2)
    sy=-np.sin(t1)
    sz=-np.sin(t2)
    maty=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    matz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return np.dot(maty,matz)

#%%
def ellips_fit_total(deformation,points):
    rot_points=trans_vec(points)
    ini_def=[*deformation,0,0]
    ## the ellipse 
    con = ({'type': 'ineq', 'fun': lambda x:  x[0]-0.05},{'type': 'ineq', 'fun': lambda x:  x[1]-0.05},{'type': 'ineq', 'fun': lambda x:  x[2]-0.05},{'type': 'ineq', 'fun': lambda x:  x[3]},{'type': 'ineq', 'fun': lambda x:  -x[3]+np.pi},{'type': 'ineq', 'fun': lambda x:  x[4]},{'type': 'ineq', 'fun': lambda x:  -x[4]+np.pi})

    fun = lambda x: np.sum([((np.dot(rot_mat(x[3],x[4]),tp)[0]**2/x[0]**2+np.dot(rot_mat(x[3],x[4]),tp)[1]**2/x[1]**2+np.dot(rot_mat(x[3],x[4]),tp)[2]**2/x[2]**2)-1)**2 for tp in rot_points])
    opts={'maxiter':1e6, 'eps':0.00001,'ftol':0.0000001}
    ### run minimize
    fit_model=scipy.optimize.minimize(fun , ini_def,constraints=con,options=opts)
    return fit_model

#%%
## plotting the fitted ellipse
def fitlipse(defo):
    t1=defo[3]
    t2=defo[4]
    defo=[defo[0],defo[1],defo[2]]
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = defo[0]*sin(phi)*cos(theta)
    y = defo[1]*sin(phi)*sin(theta)
    z = defo[2]*cos(phi)
    rot_coords= [x,y,z]
    if(t1!=0 or t2!=0):
        rot_coords=[ np.dot(rot_mat(-t1,-t2),[xt,yt,zt]) for xt,yt,zt in zip(x,y,z)]
        rot_coords=np.swapaxes(rot_coords,0,1)
    return rot_coords


#%%
"""Fit and plot"""
## fit data is saved in datatable.txt in a format suitabel for LaTeX compliation
my_cmap = sns.light_palette("Navy", as_cmap=True)
col1=my_cmap(200)
col2=my_cmap(50)
with open('datatable.txt', 'w') as f:
    datastring=" \\toprule   deformation parameters & expected axes & best fit axes & angle of axis & least squares / d.o.f. \\\\  \midrule \n "
    f.write(datastring)
    for index,da in df.iterrows():
        ## plot just the data
        ## rotate first to get better initial fit
        fig = plt.figure(figsize=(6,6),constrained_layout=True)
        ax = fig.add_subplot(1,1,1, projection='3d')
        da_rot=trans_vec(da.emb)
        ax.scatter(da_rot[:,0],da_rot[:,1],zs=da_rot[:,2],color=col1) 
        ## fit the data     
        data=ellips_fit_total(da.defo,da.emb)
        print(data)
        temp_def=data['x']
        fun_v=data['fun']
        ## plot the fit 
        temp_def=[temp_def[0],temp_def[1],temp_def[2],temp_def[3],temp_def[4]]
        x1,y1,z1= fitlipse(temp_def)
        ax.plot_surface(x1, y1, z1,  rstride=1, cstride=1, alpha=0.2, color=col2,shade=True,linewidth=0,edgecolors=col2)
    
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_box_aspect([1,1,1])
        plt.tight_layout()
        plot_filename="My_fit.pdf"
        plt.savefig(plot_filename,bbox_inches='tight')
        ### sorting to make comparisons easier
        da.defo.sort()
        fits=temp_def[:3]
        fits.sort()
        est=[1/(da.defo[0]*da.defo[1]),1/(da.defo[0]*da.defo[2]),1/(da.defo[1]*da.defo[2])]
        est.sort()
        datastring="$ ({:.2f},{:.2f},{:.2f})$ & $ ({:.2f},{:.2f},{:.2f})$ &$ ({:.2f},{:.2f},{:.2f})$ & $ ({:.2f},{:.2f})$ &$ {:.4f}  $  \\\\ \n ".format(*da.defo,*est,*fits,temp_def[3],temp_def[4],fun_v/(da.np-len(data['x'])))
        f.write(datastring)
        plt.show()
    f.write("\\bottomrule")
# %%

### close the data
for d in data:
    d.close()
