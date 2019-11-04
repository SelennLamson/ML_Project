#-*-coding:Utf-8 -*
"""Computes the SVD of rating matrix R with scipy.sparse.linalg.svds, which is memory-constant."""

import numpy as np
import scipy.sparse.linalg as slinalg
from scipy.sparse import csc_matrix
import os

# Reading and importing ratings matrix
print("Reading rating matrix R...")
import dataIO.ratingsRead
R = dataIO.ratingsRead.ratings.astype(float)

print("Performing sparse SVD...")
k = 5000
U, S, Vt = slinalg.svds(R, k=k)

print("Terminated.")

print("U:", U.shape)
print("S:", S.shape)
print("Vt:", Vt.shape)

path_svd = "trainedModels/svd_" + str(k) + ".npz"
np.savez(path_svd, U=U, S=S, Vt=Vt)
