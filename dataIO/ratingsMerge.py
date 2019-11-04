#-*-coding:Utf-8 -*
"""Launch this script to merge two rating imports with eachother."""

import numpy as np
import os
import pickle
import re

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
	ratings1 = np.load(base + f1 + path_ratings)

	# Reading folder n°2
	users2 = pickle.load(open(base + f2 + path_users, 'rb'))
	animes2 = pickle.load(open(base + f2 + path_animes, 'rb'))
	ratings2 = np.load(base + f2 + path_ratings)

	# Merging data from n°2 into n°1

	# -----------------------------------------------------------------
	# ------------------------ TODO: CODE HERE ------------------------
	# -----------------------------------------------------------------

	# Saving new data in output folder
	os.mkdir(base + output)
	pickle.dump(users1, open(base + output + path_users, 'wb'))
	pickle.dump(animes1, open(base + output + path_animes, 'wb'))
	np.save(base + output + path_ratings, ratings1)

	print("Data merged successfully. You can safely delete folders: '{}' and '{}'".format(f1, f2))

except IOError as e:
	print('Error with files:\n', e)