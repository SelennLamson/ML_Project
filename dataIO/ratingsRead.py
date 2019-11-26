#-*-coding:Utf-8 -*
"""Once rating data has been imported (script ratingsImport), import this script to read the ratings matrix R."""

import numpy as np
import os
import pickle


def read(path):
	"""Path should be something like '../TreatedData/60_to_65'."""

	path_users = path + '/users.pkl'
	path_animes = path + '/animes.pkl'
	path_sparse_ratings = path + '/ratings.pkl'

	# The matrix containing all ratings: rows are users, cols are series, cells are 0/10 ratings
	ratings = None
	# The list of user names, in the order they appear in the rating matrix
	users = []
	# The list of series ids, in the order they appear in the rating matrix (which is sorted, but not dense)
	series = []

	if os.path.exists(path_users):
		if os.path.exists(path_animes):
			if os.path.exists(path_sparse_ratings):
				with open(path_users, 'rb') as f:
					users = pickle.load(f)
				with open(path_animes, 'rb') as f:
					series = pickle.load(f)
				with open(path_sparse_ratings, "rb") as f:
					ratings = pickle.load(f)

	return users, series, ratings
