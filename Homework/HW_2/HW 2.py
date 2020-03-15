# ECE471/571 HW 2
# Ken Boling
# Python 3.6.2
# Pycharm Development Environment
# Following code was modified from the rewritten twomodal.m matlab code snippets

#Import modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis
#set output directory for figures etc.

import scipy
from scipy import spatial


outputdir = 'C:\\Users\\Ken\\Google Drive\\UTK Class work\\Pattern Classification\\HW 2'

#import data from file into a pandas dataframe
hw2_data = pd.read_csv('C:\\Users\\Ken\\Google Drive\\UTK Class work\\Pattern Classification\\HW 2\\hw2_data.csv')

x = np.array([[0.85], [1.15]])
d = 2
mu1 = np.array([[1.0], [1.375]])
mu2 = np.array([[0.7], [1.025]])
S1 = np.array([[(.1/3), (0.05/3)], [(0.05/3), (.475/3)]])
S2 = np.array([[(.025/3), 0], [0, (.025/3)]])

#set up range of values to calcualte over
ix = np.arange(start=0, stop=2.25,step=0.01 )
jy = np.arange(start=0, stop=2.25,step=0.01 )

#create an empty array of the size required by the values entered above

pxclass1=np.zeros([np.size(jy),np.size(ix)])
pxclass2=np.zeros([np.size(jy),np.size(ix)])

#Iterate over the empty array calculating the result for each cell in the px array individually.
# ni and nj act as counters representing the number of loops for each iteration and are used to define the x,y coordinates for each cell
for i, ni in enumerate(ix):
    for j, nj in enumerate(jy):
        x = np.array([[ni],[nj]])
        px1 = 1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S1) ** (1 / 2)) * np.exp(
            (-1 / 2) * (x - mu1).conj().transpose() @ np.linalg.inv(S1) @ (x - mu1))

        px2 = 1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S2) ** (1 / 2)) * np.exp(
            (-1 / 2) * (x - mu2).conj().transpose() @ np.linalg.inv(S2) @ (x - mu2))

        pxclass1[j,i]= px1
        pxclass2[j,i]= px2


#plot the figures

#Sets up the color definitions for the datapoints (this code only works for two classes, there is probably a smarter way to do this...)
colordef = ['red' if i == 1 else 'green' for i in hw2_data['class']]

#plots the data
figc1, ax = plt.subplots(figsize=(10, 10))
contourvaluesclass1 = ax.contour(ix, jy, pxclass1)
ax.clabel(contourvaluesclass1, inline=1, fontsize=10)
ax.scatter(x=hw2_data['x'], y=hw2_data['y'],c=colordef ,label=hw2_data['class'])
ax.grid()
ax.set_title ('HW 2 class 1')
plt.show()
figc1.savefig((os.path.join(outputdir, 'HW 2 1b plot1') + '.png'), dpi=600)


#plots the data
figc2, ax = plt.subplots(figsize=(10, 10))
contourvaluesclass2 = ax.contour(ix, jy, pxclass2)
ax.clabel(contourvaluesclass2, inline=1, fontsize=10)
ax.scatter(x=hw2_data['x'], y=hw2_data['y'],c=colordef ,label=hw2_data['class'])
ax.grid()
ax.set_title ('HW 2 class 2')
plt.show()
figc2.savefig((os.path.join(outputdir, 'HW 2 1b plot2 test') + '.png'), dpi=600)


# plot the result in 3D
#X3d, Y3d = np.meshgrid(ix, jy)
#fig3d = plt.figure()
#ax = Axes3D(fig3d)
#ax.plot_surface(X3d, Y3d, pxclass1, rstride=1, cstride=1, cmap=cm.coolwarm)
#ax.plot_surface(X3d, Y3d, pxclass2, rstride=1, cstride=1, cmap=cm.viridis)

#plt.show()
#fig3d.savefig((os.path.join(outputdir, 'HW 2 3d plot test') + '.png'), dpi=600)

d1 = scipy.spatial.distance.mahalanobis(u=mu1, v=x, VI=(np.linalg.inv(S1)) )
d2 = scipy.spatial.distance.mahalanobis(u=mu2, v=x, VI=(np.linalg.inv(S2)) )

print (d1, d2)