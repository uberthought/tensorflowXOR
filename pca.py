import csv
import tensorflow as tf
import numpy as np
import os.path
import math

import matplotlib

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pca(X, Y):
    mean_vec = np.mean(X, axis=0)
    cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)

    cov_mat = np.cov(X.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    for i in eig_pairs:
        print(i[0])

    matrix_w = np.hstack((eig_pairs[0][1].reshape(len(eig_pairs[0][1]),1),
                          eig_pairs[1][1].reshape(len(eig_pairs[0][1]),1),
                          eig_pairs[2][1].reshape(len(eig_pairs[0][1]),1),
                          eig_pairs[3][1].reshape(len(eig_pairs[0][1]),1),
                          eig_pairs[4][1].reshape(len(eig_pairs[0][1]),1)))

    cov_X = X.dot(matrix_w)

    # plot pca results

    x = cov_X[:, 0]
    y = cov_X[:, 1]
    z = cov_X[:, 2]
    colors = Y[:, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=x, ys=y, zs=z, c=colors)
    plt.show()
