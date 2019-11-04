#-*-coding:Utf-8 -*
"""Once rating data has been imported (script ratingsImport), import this script to read the ratings matrix R."""

import numpy as np
import os
import pickle

# -----------------------------------------
#        IMPORT THOSE VARIABLES:
# The matrix containing all ratings: rows are users, cols are series, cells are 0/10 ratings
ratings = np.zeros((0, 0), dtype=np.int8)
# The list of user names, in the order they appear in the rating matrix
users = []
# The list of series ids, in the order they appear in the rating matrix (which is sorted, but not dense)
series = []
# -----------------------------------------

path_users = '../TreatedData/users.pkl'
path_animes = '../TreatedData/animes.pkl'
path_sparse_ratings = '../TreatedData/sparse_ratings.pkl'

if os.path.exists(path_users):
	if os.path.exists(path_animes):
		if os.path.exists(path_sparse_ratings):
			with open(path_users, 'rb') as f:
				users = pickle.load(f)
			with open(path_animes, 'rb') as f:
				series = pickle.load(f)
			with open(path_sparse_ratings, "rb") as f:
				ratings = pickle.load(f)
