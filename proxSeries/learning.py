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


def ratingDistance(itemA, itemB):
	"""Computes the rating distance between series at col itemA and at col itemB"""
	colA = R[:, itemA]
	colB = R[:, itemB]

	# Masking any element from colA or colB that is not != 0 in both
	colA *= np.sign(colB)**2
	colB *= np.sign(colA)**2

	n = np.count_nonzero(colA)
	return sum(np.abs(colA - colB)) / n


def outputDistance(alphaA, alphaB):
	"""Computes the L1-Norm between vectors alphaA and alphaB, normalizing by their dimension"""
	return sum(np.abs(alphaA - alphaB)) / l


def distLoss(itemA, itemB, alphaA, alphaB):
	"""Computes the loss for items A and B, and their output from the model
	Output is three elements:
		the loss value (scalar),
		the derivative for gammas (matrix),
		the derivative for betas (matrix)"""
	diff = ratingDistance(itemA, itemB) - outputDistance(alphaA, alphaB)
	loss = diff**2
	gammaDeriv = np.zeros((l, p))
	#gammaDeriv = (-2/l) * diff * np.sign(alphaA - alphaB)
	betaDeriv = np.zeros((l, p))
	return loss, gammaDeriv, betaDeriv


# ------------------------------------------------------
#                 MODEL INITIALIZATION

# Model: f(X) = Gamma.X² + Beta.X
# X:      p x 1 -> series information (p ~= 100)
# f(X):   l x 1 -> series representation (l < p)
# Gamma:  l x p -> coefficients of power 2
# Beta:   l x p -> coefficients of power 1

p = 100
l = 10
Gamma = np.random.rand((l, p))
Beta = np.random.rand((l, p))

# ------------------------------------------------------
#                 MODEL INITIALIZATION