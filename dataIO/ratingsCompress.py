#-*-coding:Utf-8 -*
"""Data validation, to be sure that rating data was imported correctly"""

import numpy as np
import os
import pickle
from scipy.sparse import csc_matrix

data = np.zeros((0, 0), dtype=np.int8)

path_ratings = '../TreatedData/ratings.npy'
path_sparse_ratings = '../TreatedData/sparse_ratings.pkl'

print("Loading uncompressed data...")
data = np.load(path_ratings)
print("Converting to sparse matrix...")
sparse_data = csc_matrix(data, dtype=int)

print("Saving compressed matrix...")
with open(path_sparse_ratings, "wb") as file:
	pickle.dump(sparse_data, file)

print("Terminated.")

#hello