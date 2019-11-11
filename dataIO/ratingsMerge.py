#-*-coding:Utf-8 -*
"""Launch this script to merge two rating imports with eachother."""

import numpy as np
import os
import pickle
import re
from scipy.sparse import csc_matrix


def sorted_search(ar, x, get_closest=False):#ar array x 
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


def sorted_insert(ar, x):
	ind = sorted_search(ar, x, True)
	ar.insert(ind, x)
	return ind

def get_coomatrix(A):
	rows=A.shape[0]
	columns=A.shape[1]
	row=[]
	column=[]
	value=[]
	for i in range(rows):
		for j in range(columns):
			if A[i][j]!=0:
				row.append(i)
				column.append(j)
				value.append(A[i][j])
	return row,column,value


base = '../TreatedData/'
folders = os.listdir(base)
folders = [f for f in folders if re.match(r'[0-9]+_to_[0-9]+', f)]

for i in range(len(folders)):
	print("[{}] - folder '{}'".format(i, folders[i]))

while True:
	f1 = input("Please select FIRST folder: ")
	try:
		f1 = int(f1)
		if 0 <= f1 < len(folders):
			break
	except ValueError:
		continue

while True:
	f2 = input("Please select SECOND folder: ")
	try:
		f2 = int(f2)
		if 0 <= f2 < len(folders) and f2 != f1:
			break
	except ValueError:
		continue

f1 = folders[f1]
f2 = folders[f2]

print("\nMerging folders", f1, "and", f2, "...")

val11 = int(f1.split('_')[0])
val12 = int(f1.split('_')[2])
val21 = int(f2.split('_')[0])
val22 = int(f2.split('_')[2])
valmin = min(val11, val21)
valmax = max(val21, val22)
joined = val11 <= val21 <= val12 or val21 <= val11 <= val22
output = (str(valmin) + '_to_' + str(valmax)) if joined else ('merged_' + f1 + '_and_' + f2)

print("Output folder will be:", output, "\n")

try:
	path_users = '/users.pkl'
	path_animes = '/animes.pkl'
	path_ratings = '/ratings.npy'
	path_info = '/info.pkl'

	# Reading folder n°1
	users1 = pickle.load(open(base + f1 + path_users, 'rb'))
	animes1 = pickle.load(open(base + f1 + path_animes, 'rb'))
	ratings1 = pickle.load(open(base + f1 + path_ratings, 'rb'))
	# Reading folder n°2
	users2 = pickle.load(open(base + f2 + path_users, 'rb'))
	animes2 = pickle.load(open(base + f2 + path_animes, 'rb'))
	ratings2 = pickle.load(open(base + f2 + path_ratings, 'rb'))
	# We are not converting ratings2 as a dense matrix, we will use it as such

	# Choosing the biggest anime dataset as the base matrix, to avoid wasting time on animes merging (the longest)
	if len(animes2) > len(animes1):
		users1, users2 = users2, users1
		animes1, animes2 = animes2, animes1
		ratings1, ratings2 = ratings2, ratings1
	ratings1 = ratings1.toarray()

	# Extending ratings1 to contain every new user and anime of ratings2
	anime_mapping = []	# Index of ratings2's animes in new extended matrix
	users_mapping = []	# Index of ratings2's users in new extended matrix

	# Adding a new line for each new user, or remembering its index if already there
	print("\n--- Merging users ---")
	total_users = len(users2)
	it = progress = 0

	to_add = 0
	for user in users2:
		if user in users1:
			users_mapping.append(users1.index(user))
		else:
			users_mapping.append(len(users1))
			users1.append(user)
			to_add += 1

		it += 1
		percent = it / total_users
		if int(percent * 10) * 10 > progress:
			progress = round(percent * 10) * 10
			print('Merging users: {}%'.format(progress))

	ratings1 = np.vstack([ratings1, np.zeros((to_add, len(animes1)))])

	# Inserting a new column at the sorted location if anime unknown, or remembering its index if already there
	# This part takes into account that animes are always sorted by id, so we can reduce the search field each time
	print("\n--- Merging animes ---")
	total_animes = len(animes2)
	it = progress = 0
	current_index = 0
	for anime in animes2:

		it += 1
		percent = it / total_animes
		if int(percent * 10) * 10 > progress:
			progress = round(percent * 10) * 10
			print('Merging animes: {}%'.format(progress))

		while current_index < len(animes1) and animes1[current_index] < anime:
			current_index += 1
		if current_index == len(animes1):
			animes1.append(anime)
			ratings1 = np.hstack([ratings1, np.zeros((len(users1), 1))])
			current_index += 1
		elif animes1[current_index] == anime:
			anime_mapping.append(current_index)
		else:
			animes1.insert(current_index, anime)
			ratings1 = np.hstack([ratings1[:, :current_index], np.zeros((len(users1), 1)), ratings1[:, current_index:]])
			anime_mapping.append(current_index)
		current_index += 1


	# reduced_animes = animes1
	# previous_index = 0
	# for anime in animes2:
	# 	if anime in reduced_animes:
	# 		index = sorted_search(reduced_animes, anime) + previous_index
	# 		anime_mapping.append(index)
	#
	# 		reduced_animes = animes1[index + 1:]
	# 		previous_index = index + 1
	# 	else:
	# 		new_index = sorted_search(reduced_animes, anime, get_closest=True) + previous_index
	# 		anime_mapping.append(new_index)
	#
	# 		animes1.insert(new_index, anime)
	# 		ratings1 = np.hstack([ratings1[:, :new_index], np.zeros((len(users1), 1)), ratings1[:, new_index:]])
	#
	# 		reduced_animes = animes1[new_index + 1:]
	# 		previous_index = new_index + 1
	#
	# 	it += 1
	# 	percent = it / total_animes
	# 	if int(percent * 10) * 10 > progress:
	# 		progress = round(percent * 10) * 10
	# 		print('Merging animes: {}%'.format(progress))

	# Merge the values of ratings2 into ratings1, using the sparse matrix representation to have all non-zero values
	print("\n--- Merging values ---")
	coomat = ratings2.tocoo()
	n_values = len(coomat.data)
	it = progress = 0
	for row, col, val in zip(coomat.row, coomat.col, coomat.data):
		new_row = users_mapping[row]
		new_col = anime_mapping[col]
		ratings1[new_row, new_col] = val

		it += 1
		percent = it / n_values
		if int(percent * 10) * 10 > progress:
			progress = round(percent * 10) * 10
			print("Merging values: {}%".format(progress))

	# Saving new data in output folder
	os.mkdir(base + output)
	pickle.dump(users1, open(base + output + path_users, 'wb'))
	pickle.dump(animes1, open(base + output + path_animes, 'wb'))
	pickle.dump(csc_matrix(ratings1, dtype=int), open(base + output + path_ratings, 'wb'))
	print("Data merged successfully.")
except IOError as e:
	print('Error with files:\n', e)


