#-*-coding:Utf-8 -*
"""Data validation, to be sure that rating data was imported correctly"""

import numpy as np
import os
import pickle


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

path_data = '../Data/UserAnimeList.csv'
path_users = '../TreatedData/users.pkl'
path_animes = '../TreatedData/animes.pkl'
path_ratings = '../TreatedData/ratings.npy'
path_info = '../TreatedData/info.pkl'

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


with open(path_data, "r", encoding="utf8") as file:
	file.readline()  # Headers
	line_index = 1
	accurate = 0
	percent = 0

	for line in file:
		line_index += 1

		try:
			elts = line.split(",")
			if len(elts) < 6:
				accurate += 1
				continue

			username = elts[0]
			anime_id = int(elts[1])
			score = elts[5]

			row = column = 0

			try:
				row = users.index(username)
				col = animes.index(anime_id)
				assert data[row, col] == int(score)
				accurate += 1
			except (ValueError, AssertionError):
				if print_errors:
					print("Found error on line", line_index)
					print(username, anime_id, score)
		except Exception as e:
			print("ERROR: Incorrect data point, line " + str(line_index) + ". Didn't stop data import.", e)
			accurate += 1

		if line_index > checks or line_index >= skip_lines:
			break
		if round(line_index / checks * 100) > percent:
			percent = round(line_index / checks * 100)
			print("Data validation... {}% - accurate {}%".format(percent, round(accurate / line_index * 100)))

print("Validating:", accurate, "accurate data points over", line_index - 1, "-", round(accurate / line_index * 100), "%")
