#-*-coding:Utf-8 -*
"""Launch this script to import series data-set into a preprocessed data-set with "linear" numbers only."""

import numpy as np
import os
import pickle
import csv


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


def typeConvert(type):
	if type == 'TV':		return 6
	if type == 'Movie':		return 5
	if type == 'Special':	return 4
	if type == 'OVA':		return 3
	if type == 'ONA':		return 2
	if type == 'Music':		return 1
	else:					return 0


def sourceConvert(source):
	if source == 'Novel':			return 15
	if source == 'Book':			return 14
	if source == 'Manga':			return 13
	if source == 'Digital manga':	return 12
	if source == 'Web manga':		return 11
	if source == 'Original':		return 10
	if source == 'Visual novel':	return 9
	if source == 'Game':			return 8
	if source == 'Light novel':		return 7
	if source == '4 - koma manga':	return 6
	if source == 'Card game':		return 5
	if source == 'Picture book':	return 4
	if source == 'Music':			return 3
	if source == 'Radio':			return 2
	if source == 'Other':			return 1
	else:							return 0


def statusConvert(status):
	if status == 'Not yet aired':		return 0
	if status == 'Currently Airing':	return 1
	if status == 'Finished Airing':		return 2
	else:								return 0


def durationConvert(duration):
	minutes = 0
	elts = duration.split(" ")

	for i in range(len(elts) - 1):
		try:
			val = int(elts[i])
			if elts[i+1] == "min.":
				minutes += val
			elif elts[i+1] == "hr.":
				minutes += val * 60
			elif elts[i+1] == "sec.":
				minutes += val / 60
		except ValueError:
			continue

	return minutes


def ratingConvert(rating):
	if rating == 'Rx - Hentai':						return 6
	if rating == 'R+ - Mild Nudity':				return 5
	if rating == 'R - 17+ (violence & profanity)':	return 4
	if rating == 'PG-13 - Teens 13 or older':		return 3
	if rating == 'G - All Ages':					return 2
	if rating == 'PG - Children':					return 1
	else:											return 0


def premieredConvert(premiered):
	elts = premiered.split(" ")
	if len(elts) != 2:
		return 0
	date = int(elts[1]) - 2000
	if elts[0] == 'Winter':	date += 0
	if elts[0] == 'Spring':	date += 0.25
	if elts[0] == 'Summer':	date += 0.5
	if elts[0] == 'Fall':	date += 0.75
	return date


def broadcastConvert(broadcast):
	day = broadcast.split(" ")[0]
	if day == 'Mondays':		return 1
	if day == 'Tuesdays':		return 2
	if day == 'Wednesdays':		return 3
	if day == 'Thursdays':		return 4
	if day == 'Fridays':		return 5
	if day == 'Saturdays':		return 6
	if day == 'Sundays':		return 7
	else:						return 0


series = np.zeros((0, 10), dtype=int)

path_data = '../Data/AnimeList.csv'
path_series = '../TreatedData/series.npy'

accumulated_size = 0

with open(path_data, "r", encoding="utf8") as file:
	csvfile = csv.reader(file, delimiter=',', quotechar='"')
	line_index = 0

	total_size = os.path.getsize(path_data)
	percent = 0

	for line in csvfile:
		line_index += 1
		accumulated_size += sum([len(elt) for elt in line])
		if round(accumulated_size / total_size * 100) > percent:
			percent = round(accumulated_size / total_size * 100)
			print("Data import... {}% - line {}".format(percent, line_index))
		if line_index == 0:
			continue

		try:
			if len(line) != 31:
				continue

			s_id = int(line[0])
			s_type = typeConvert(line[6])
			source = sourceConvert(line[7])
			try:
				episodes = int(line[8])
			except ValueError:
				episodes = 0
			status = statusConvert(line[9])
			duration = durationConvert(line[13])
			rating = ratingConvert(line[14])
			premiered = premieredConvert(line[22])
			broadcast = broadcastConvert(line[23])
			views = int(line[19])

			series = np.vstack((series, np.array([s_id, s_type, source, episodes, status, duration, rating, premiered, broadcast, views])))
		except Exception as e:
			print("ERROR: Incorrect data, line " + str(line_index) + ". Didn't stop data import.", e)

print("Finishing importing", series.shape[0], "series.")
np.save(path_series, series)
