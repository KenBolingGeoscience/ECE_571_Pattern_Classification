# ECE471/571 project 1
# Ken Boling
# Python 3.6.2
# Pycharm Development Environment
# Following code was rewritten from the plotsyth.m and twomodal.m matlab code snippets

# Import modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time


# set output directory for figures etc.
outputdir = 'C:\\Users\\Ken\\Google Drive\\UTK Class work\\Pattern Classification\\project 1'

# import synth.tr training data from file into a pandas dataframe
trdata = pd.read_csv('C:\\Users\\Ken\\Google Drive\\UTK Class work\\Pattern Classification\\project 1\\synth.tr',
                     delim_whitespace=True)

# check if data loaded correctly
print(trdata.info(), trdata.head())
# looks good

# assign values as in the twomodal.m example, numpy ndarrys are used instead of matrices since they work the same way and are apparently "better" according to:
# https://www.numpy.org/devdocs/user/numpy-for-matlab-users.html
d_1 = 2
mu1_1 = np.array([[-0.75], [0.2]])
mu2_1 = np.array([[0.3], [0.3]])
S1_1 = np.array([[0.25, 0], [0, 0.3]])
S2_1 = np.array([[0.1, 0], [0, 0.1]])
A1_1 = 0.8
A2_1 = 1 - A1_1

d_2 = 2
mu1_2 = np.array([[-0.45], [0.75]])
mu2_2 = np.array([[0.5], [0.6]])
S1_2 = np.array([[0.3, 0], [0, 0.4]])
S2_2 = np.array([[0.1, 0], [0, 0.1]])
A1_2 = 0.8
A2_2 = 1 - A1_2

# set up range of values to calcualte over
ix = np.arange(start=-1.5, stop=1, step=0.01)
jy = np.arange(start=-0.2, stop=1, step=0.01)

ix_2 = np.arange(start=-1.5, stop=1, step=0.01)
jy_2 = np.arange(start=-0.2, stop=1, step=0.01)


def two_modal_gaussian(d, mu1, mu2, S1, S2, A1, A2, ix, jy, title ):
    # create an empty array of the size required by the values entered above
    px = np.zeros([np.size(jy), np.size(ix)])
    # Iterate over the empty array calculating the result for each cell in the px array individually.
    # This took awhile to figure out, but I learned the utility of using "enumerate" in a for loop in the process!
    # ni and nj act as counters representing the number of loops for each iteration and are used to define the x,y coordinates for each cell
    for i, ni in enumerate(ix):
        for j, nj in enumerate(jy):
            x = np.array([[ni], [nj]])
            px1 = A1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S1) ** (1 / 2)) * np.exp(
                (-1 / 2) * (x - mu1).conj().transpose() @ np.linalg.inv(S1) @ (x - mu1))

            px2 = A2 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S2) ** (1 / 2)) * np.exp(
                (-1 / 2) * (x - mu2).conj().transpose() @ np.linalg.inv(S2) @ (x - mu2))

            px[j, i] = px1 + px2
    # plot the figures

    # Sets up the color definitions for the datapoints (this code only works for two classes, there is probably a smarter way to do this...)
    colordef = ['red' if i == 0 else 'green' for i in trdata['yc']]

    # plots the data
    fig1, ax = plt.subplots()
    contourvalues = ax.contour(ix, jy, px)
    ax.clabel(contourvalues, inline=1, fontsize=10)
    ax.scatter(x=trdata['xs'], y=trdata['ys'], c=colordef, label=trdata['yc'])
    ax.grid()
    ax.set_title(title)
    plt.show()
    fig1.savefig((os.path.join(outputdir, title) + '.png'), dpi=600)

    # plot the result in 3D
    X3d, Y3d = np.meshgrid(ix, jy)
    fig2 = plt.figure()
    ax = Axes3D(fig2)
    ax.plot_surface(X3d, Y3d, px, rstride=1, cstride=1, cmap=cm.coolwarm)
    plt.show()
    fig1.savefig((os.path.join(outputdir, (title + '_3D') + '.png')))
    return px

class_0_two_modal = two_modal_gaussian(
    d=d_1,
    mu1=mu1_1,
    mu2=mu2_1,
    S1=S1_1,
    S2=S2_1,
    A1=A1_1,
    A2=A2_1,
    ix=ix, jy=jy, title='Training Data class 0')

class_1_two_modal = two_modal_gaussian(
    d=d_2,
    mu1=mu1_2,
    mu2=mu2_2,
    S1=S1_2,
    S2=S2_2,
    A1=A1_2,
    A2=A2_2,
    ix=ix, jy=jy, title='Training Data class 1')


# Case 3 (Maximum A-Priori Probability, Mahalanobis Distance)

# Define the Case 3 function:

def two_modal_gaussian_case_33(X,  d_0, mu1_0, mu2_0, S1_0, S2_0, A1_0, A2_0, d_1, mu1_1, mu2_1, S1_1, S2_1, A1_1, A2_1, ix, jy, title,):
    start = time.time()  # timer started
    xrow = X[X.columns[:2]]
    mode_1_class_0 = A1_0 / (((2 * np.pi) ** (d_0 / 2)) * np.linalg.det(S1_0) ** (1 / 2)) * np.exp(
        (-1 / 2) * (xrow - mu1_0).conj().transpose() @ np.linalg.inv(S1_0) @ (xrow - mu1_0))

    mode_2_class_0 = A2_0 / (((2 * np.pi) ** (d_0 / 2)) * np.linalg.det(S2_0) ** (1 / 2)) * np.exp(
        (-1 / 2) * (xrow - mu2_0).conj().transpose() @ np.linalg.inv(S2_0) @ (xrow - mu2_0))

    mode_1_class_1 = A1_1 / (((2 * np.pi) ** (d_1 / 2)) * np.linalg.det(S1_1) ** (1 / 2)) * np.exp(
        (-1 / 2) * (xrow - mu1_1).conj().transpose() @ np.linalg.inv(S1_1) @ (xrow - mu1_1))

    mode_2_class_1 = A2_1 / (((2 * np.pi) ** (d_1 / 2)) * np.linalg.det(S2_1) ** (1 / 2)) * np.exp(
        (-1 / 2) * (xrow - mu2_1).conj().transpose() @ np.linalg.inv(S2_1) @ (xrow - mu2_1))

    #g1 = -md1 / 2 - ((np.log(np.linalg.det(S1))) / 2)
    #g2 = -md2 / 2 - ((np.log(np.linalg.det(S2))) / 2)

    # md1 = np.sqrt(
    # np.dot(
    #                np.dot((xrow - muv1).T , np.linalg.inv(E1))
    #                   ,(xrow - muv1)))
    # md2 = np.sqrt(
    #            np.dot(
    #                np.dot((xrow - muv2).T , np.linalg.inv(E2))
    #                    ,(xrow - muv2)))
  #  conditions = ((mode_1_class_0 | mode_2_class_0) > (mode_1_class_0 | mode_2_class_0)  g1 > g2, g1 < g2)
   # choices = (0, 1)

    #xy = np.array(X.iloc[:, :2])  # select all rows in the first 2 columns




    #case_2_class = []  # emtpy data object
    #for irow in xy:
        #xrow = irow.reshape(-1, 1)  # converts the row vector to a column vector
        #test1 = (xrow - muv1)
        # print(test1)

        # Calculates the mahalanobis distance between each data point and the mean determined for each class

      #  mode_1_class_0  = A1_1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S1) ** (1 / 2)) * np.exp(
       #         (-1 / 2) * (xrow - mu1).conj().transpose() @ np.linalg.inv(S1) @ (xrow - mu1))

       # mode_2_class_0  = A2_1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S2) ** (1 / 2)) * np.exp(
    #        (-1 / 2) * (xrow - mu2).conj().transpose() @ np.linalg.inv(S2) @ (xrow - mu2))

       # mode_1_class_1 = A1 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S1) ** (1 / 2)) * np.exp(
      #      (-1 / 2) * (xrow - mu1).conj().transpose() @ np.linalg.inv(S1) @ (xrow - mu1))

     #   mode_2_class_1 = A2 / (((2 * np.pi) ** (d / 2)) * np.linalg.det(S2) ** (1 / 2)) * np.exp(
    #        (-1 / 2) * (xrow - mu2).conj().transpose() @ np.linalg.inv(S2) @ (xrow - mu2))

     #   g1 = -md1 / 2 - ((np.log(np.linalg.det(S1))) / 2)
      #  g2 = -md2 / 2 - ((np.log(np.linalg.det(S2))) / 2)

        # md1 = np.sqrt(
        # np.dot(
        #                np.dot((xrow - muv1).T , np.linalg.inv(E1))
        #                   ,(xrow - muv1)))
        # md2 = np.sqrt(
        #            np.dot(
        #                np.dot((xrow - muv2).T , np.linalg.inv(E2))
        #                    ,(xrow - muv2)))
       # conditions = (g1 > g2, g1 < g2)
      #  choices = (0, 1)
      #  select = np.select(conditions, choices)
      #  case_2_class.append(select)

    #discrim = pd.DataFrame(np.concatenate(case_2_class))
   # discrim.rename(columns={0: 'case_2_class'}, inplace=True)

   # Xout = pd.concat([X, discrim], axis=1)

    # prints
    #end = time.time()
    #print('time to complete:', end - start, 'seconds')
    #return Xout





def two_modal_gaussian_case_3(df, d_0, mu1_0, mu2_0, S1_0, S2_0, mu1_1, mu2_1, S1_1, S2_1, ix, jy, title, ):
    start = time.time()  # timer started
    xrow = df[df.columns[:2]]

    mode_1_class_0 = np.dot(np.dot((xrow - mu1_0).T, np.linalg.inv(S1_0)), (xrow - mu1_0))

    mode_2_class_0 = np.dot(np.dot((xrow - mu2_0).T, np.linalg.inv(S2_0)), (xrow - mu2_0))

    mode_1_class_1 = np.dot(np.dot((xrow - mu1_1).T, np.linalg.inv(S1_1)), (xrow - mu1_1))

    mode_2_class_1 = np.dot(np.dot((xrow - mu2_1).T, np.linalg.inv(S2_1)), (xrow - mu2_1))




    g1 = -mode_1_class_0 / 2 - ((np.log(np.linalg.det(S1_0))) / 2)
    g2 = -mode_2_class_0 / 2 - ((np.log(np.linalg.det(S2_0))) / 2)
    g3 = -mode_1_class_1 / 2 - ((np.log(np.linalg.det(S1_1))) / 2)
    g4 = -mode_2_class_1 / 2 - ((np.log(np.linalg.det(S2_1))) / 2)

    #... I know...it's only stupid if it doesn't work! right?



    conditions = ((g1 | g2) > (g3 | g4), (g1 | g2) < (g3 | g4), )
    choices = (0, 1)

    df['two_modal_gaussian_case_3'] = np.select(conditions, choices)


    # prints
    end = time.time()
    print('time to complete:', end - start, 'seconds')
    return df

two_modal_test = two_modal_gaussian_case_3(df=tedata, d_0, mu1_0, mu2_0, S1_0, S2_0, mu1_1, mu2_1, S1_1, S2_1, ix, jy, title, )

print(two_modal_test)