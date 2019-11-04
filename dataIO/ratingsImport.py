#-*-coding:Utf-8 -*
"""Launch this script with a start and end offsets to continue the import of ratings to the R matrix."""

import numpy as np
import os
import pickle

# --------------- IMPORT PARAMETERS ---------------
start_offset = 1 # start position in the file (in million lines)
end_offset = 2 # end position in the file (in million lines)
# -------------------------------------------------

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


def sorted_insert(ar, x):
	ind = sorted_search(ar, x, True)
	ar.insert(ind, x)
	return ind


def save_results(_data, _users, _animes, _line, _size):
	np.save(path_ratings, _data)
	with open(path_users, 'wb') as lf:
		pickle.dump(_users, lf)
	with open(path_animes, 'wb') as lf:
		pickle.dump(_animes, lf)
	with open(path_info, 'wb') as lf:
		pickle.dump((_line, _size), lf)
	print("---------- SAVED RESULTS ----------")


data = np.zeros((0, 0), dtype=np.int8)

users = []
animes = []
anime_titles = []

folder = '../TreatedData/' + str(start_offset) + '_to_' + str(end_offset) + '/'

if not os.path.exists(folder):
	os.mkdir(folder)

path_data = '../Data/UserAnimeList.csv'
path_users = folder + 'users.pkl'
path_animes = folder + 'animes.pkl'
path_ratings = folder + 'ratings.npy'
path_info = folder + 'info.pkl'
accumulated_size = 0
skip_lines = 0
start_offset *= 1e6
end_offset *= 1e6

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
	line_index = 0

	current_user = ""
	current_user_index = -1

	total_size = os.path.getsize(path_data)
	percent = 0
	anime_thousands = 0

	for line in file:
		line_index += 1
		if line_index <= start_offset + skip_lines:
			continue
		if line_index > end_offset:
			break

		accumulated_size += len(line)
		if round(accumulated_size / total_size * 100, 2) > percent:
			percent = round(accumulated_size / total_size * 100, 2)
			print("Data import... {}% of whole file - line {}/{}".format(
														percent,
														round(line_index - start_offset),
														round(end_offset - start_offset)))

		try:
			elts = line.split(",")
			if len(elts) < 6:
				continue

			username = elts[0]
			anime_id = int(elts[1])
			score = elts[5]

			row = column = 0

			if current_user_index != -1 and current_user == username:
				row = current_user_index
			else:
				try:
					row = users.index(username)
				except ValueError:
					row = len(users)
					users.append(username)
					data = np.vstack((data, np.zeros((1, len(animes)), dtype=np.int8)))
				current_user_index = row
				current_user = username
			try:
				column = sorted_search(animes, anime_id)
				assert column is not None
			except AssertionError:
				column = sorted_insert(animes, anime_id)
				data = np.hstack((data[:, :column], np.zeros((len(users), 1), dtype=np.int8), data[:, column:]))

				if len(animes) > (anime_thousands + 1) * 1000:
					anime_thousands += 1
					print("Discovered", anime_thousands * 1000, "animes over 15000.")

			data[row, column] = int(score)
		except Exception as e:
			print("ERROR: Incorrect data point, line " + str(line_index) + ". Didn't stop data import.", e)

		if line_index % 100000 == 0:
			save_results(data, users, animes, line_index - start_offset, accumulated_size)

print("Finishing importing", len(users), "user scores on", len(animes), "animes.")
print(round(end_offset - start_offset), "million lines imported.")

save_results(data, users, animes, line_index - start_offset, accumulated_size)

