#-*-coding:Utf-8 -*
"""Learning algorithm to produce a l-dim vector for each series, preserving distances between series
as much as possible, based on their ratings by all users."""

import numpy as np

# Reading and importing ratings matrix
import dataIO.ratingsRead
R = dataIO.ratingsRead.ratings.astype(float)
S = dataIO.ratingsRead.series
U = dataIO.ratingsRead.users

# Transforming R so that scores 1 to 10 range from -1 to 1, and 0 scores remain 0 (neutral)
R = R - 1
R = R + np.maximum(-np.sign(R), np.zeros(R.shape)) * 5.5
R = R / 4.5 - 1


def evaluateDistance(itemA, itemB):
	"""Computes the rating distance between series at col itemA and at col itemB"""
	colA = R[:, itemA]
	colB = R[:, itemB]

	# Masking any element from colA or colB that is not != 0 in both
	colA *= np.sign(colB)**2
	colB *= np.sign(colA)**2

	n = np.count_nonzero(colA)
	return sum(np.abs(colA - colB)) / n


print("Evaluating distance:")
print(evaluateDistance(0, 1))
