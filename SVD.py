#-*-coding:Utf-8 -*
"""Data validation, to be sure that data was imported correctly"""

import numpy as np
import os
import pickle
import SVDUpdate as svdu
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import scipy
from sparsesvd import sparsesvd
import cupy as cp
from irlb.irlb import tsvd


def sorted_search(ar, x, get_closest=False):
	if len(ar) == 0:
		return 0 if get_closest else None
	if len(ar) == 1:
		if ar[0] == x:
			return 0
		elif get_closest and ar[0] < x:
			return 1
		elif get_closest and ar[0] > x:
			return 0
		else:
			return None
	else:
		cutpos = len(ar) // 2
		if ar[cutpos] == x:
			return cutpos
		elif ar[cutpos] > x:
			return sorted_search(ar[:cutpos], x, get_closest)
		else:
			ind = sorted_search(ar[cutpos + 1:], x, get_closest)
			return None if ind is None else ind + cutpos + 1


data = np.zeros((0, 0), dtype=np.int8)
checks = 7880476
print_errors = False

users = []
animes = []
anime_titles = []

path_users = 'TreatedData/users.pkl'
path_animes = 'TreatedData/animes.pkl'
path_ratings = 'TreatedData/ratings.npy'
path_info = 'TreatedData/info.pkl'

if os.path.exists(path_users):
	if os.path.exists(path_animes):
		if os.path.exists(path_ratings):
			if os.path.exists(path_info):
				with open(path_info, 'rb') as f:
					skip_lines, accumulated_size = pickle.load(f)
				with open(path_users, 'rb') as f:
					users = pickle.load(f)
				with open(path_animes, 'rb') as f:
					animes = pickle.load(f)
				data = np.load(path_ratings)

print("Loaded data for", len(users), "users and", len(animes), "animes.", skip_lines, "datapoints in total.")

lines = len(users)
cols = len(animes)
D = data[:lines, :cols]
keep = round(cols * 0.1)

smat = csc_matrix(D.astype(float))  # convert to sparse CSC format
# U, S, V = sparsesvd(smat, keep)
R = tsvd(D.astype(np.float), keep, tol=0.001, maxit=20)
U, S, V = R[0], R[1], R[2]
keep = len(S)
print(U.shape, S.shape, V.shape)

reconstructed = U.dot(np.diag(S)).dot(np.transpose(V))

total = np.sum(D)
diff = np.sum(np.absolute(D - reconstructed))
print(np.linalg.norm(D - reconstructed), diff / total * 100)




# lines = 10000
# U, S, V = cp.linalg.svd(data[:lines, :])
# V = V[:lines, :]
# print(U.shape, S.shape, V.shape)
#
# newdata = U.dot(cp.diag(S)).dot(V)
#
# dist = cp.linalg.norm(data[:lines, :] - newdata)
# print(dist)

# Concepts = V[:10, :]
# Clusters = []
#
# for i in range(10):
# 	cluster = []
# 	line = list(Concepts[i, :].flatten())
# 	for j in range(10):
# 		maxind = line.index(max(line))
# 		cluster.append(registeredAnimes[maxind])
# 		line[maxind] = -1000
# 	Clusters.append(cluster)
#
# with open("Data/anime_cleaned.csv", "r", encoding="utf-8") as file:
# 	foundAnimes = 0
# 	file.readline()
#
# 	for line in file:
# 		elts = line.split(",")
# 		if len(elts) < 3:
# 			continue
# 		animeId = int(elts[0])
# 		title = elts[2]
#
# 		for i in range(len(registeredAnimes)):
# 			if registeredAnimes[i] == animeId:
# 				animeTitles[i] = title
# 				foundAnimes += 1
# 				break
#
# 		# anime_id,title,title_english,title_japanese,title_synonyms,image_url,type,source,episodes,status,airing,
# 		# aired_string,aired,duration,rating,score,scored_by,rank,popularity,members,favorites,background,premiered,
# 		# broadcast,related,producer,licensor,studio,genre,opening_theme,ending_theme,duration_min,aired_from_year
#
# for i in range(len(Clusters)):
# 	print("Cluster", i, ":", S[i])
# 	c = Clusters[i]
# 	for a in c:
# 		if a in registeredAnimes:
# 			print(animeTitles[registeredAnimes.index(a)])
# 		else:
# 			print("Unknown anime", a)
# 	print("")