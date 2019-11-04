#-*-coding:Utf-8 -*
"""Learning algorithm to produce a l-dim vector for each series, preserving distances between series
as much as possible, based on their ratings by all users."""

import numpy as np
import pickle as pkl
import os
from random import *
from notify_run import Notify
notif = Notify()


print("Reading series matrix S_data...")
S_data = np.load("../TreatedData/series.npy").astype(int)

# Reading and importing ratings matrix
print("Reading rating matrix R...")
import dataIO.ratingsRead
R = dataIO.ratingsRead.ratings.astype(float)
S = dataIO.ratingsRead.series
S = S[:R.shape[1]]
U = dataIO.ratingsRead.users

# Transforming R so that scores 1 to 10 range from -1 to 1, and 0 scores remain 0 (neutral)
print("Preprocessing rating matrix R... 1/3")
R = R - 1
print("Preprocessing rating matrix R... 2/3")
R = R + np.maximum(-np.sign(R), np.zeros(R.shape)) * 5.5
print("Preprocessing rating matrix R... 3/3")
R = R / 4.5 - 1

print("Imported rating matrix R:", R.shape, "- and series matrix S_data:", S_data.shape)


def ratingDistance(itemA, itemB):
	"""Computes the rating distance between series at col itemA and at col itemB"""
	colA = R[:, itemA] * 1
	colB = R[:, itemB] * 1

	# Masking any element from colA or colB that is not != 0 in both
	colA *= np.sign(colB)**2
	colB *= np.sign(colA)**2
	n = np.count_nonzero(colA)

	if n == 0:
		return 0
	else:
		return sum(np.abs(colA - colB)) / n


def outputDistance(alphaA, alphaB):
	"""Computes the L1-Norm between vectors alphaA and alphaB, normalizing by their dimension"""
	return sum(np.abs(alphaA - alphaB)) / l


def applyModelAndLoss(itemA, itemB, xA, xB):
	"""Computes the model output and loss for items A and B
	Output is composed of:
		the output alphaA,
		the output alphaB,
		the loss value (scalar),
		the derivative for gammas (matrix),
		the derivative for betas (matrix)"""

	# Making sure xA and xB are row vectors
	if xA.shape[0] > 1: xA = xA.T
	if xB.shape[0] > 1: xB = xB.T

	xA2 = xA**2
	xB2 = xB**2
	alphaA = Gamma @ xA2.T + Beta @ xA.T
	alphaB = Gamma @ xB2.T + Beta @ xB.T
	xDiff = xA - xB
	x2Diff = xA2 - xB2

	distDiff = (ratingDistance(itemA, itemB) - outputDistance(alphaA, alphaB))
	loss = distDiff**2

	gammaDeriv = (-2/l) * distDiff * (np.sign(Gamma * x2Diff) * x2Diff)
	betaDeriv = (-2/l) * distDiff * (np.sign(Beta * xDiff) * xDiff)

	return alphaA, alphaB, loss, gammaDeriv, betaDeriv


# ------------------------------------------------------
#                 MODEL INITIALIZATION

# Model: f(X) = Gamma.X² + Beta.X
# X:      p x 1 -> series information (p ~= 100)
# f(X):   l x 1 -> series representation (l < p)
# Gamma:  l x p -> coefficients of power 2
# Beta:   l x p -> coefficients of power 1

p = 9
l = 4
Gamma = np.random.rand(l, p)
Beta = np.random.rand(l, p)

Gamma = np.zeros((l, p))
Beta = np.zeros((l, p))

ans = input("Should we try to load previously trained model? [y, n] ")
path_model = "trainedModels/model_" + str(p) + "_to_" + str(l) + ".npz"

if ans == "y":
	if os.path.exists(path_model):
		try:
			loadedArrays = np.load(path_model)
			Gamma = loadedArrays['Gamma']
			Beta = loadedArrays['Beta']
			print("Loaded trained model for dimensions:", p, "to", l)
		except (IOError, ValueError) as e:
			print("Error loading file:", e)
	else:
		print("There is no trained model for these dimensions:", p, "to", l)


# ------------------------------------------------------
#              PAIR-WISE STOCHASTIC LEARNING

learningRate = 0.01
samplesInEpoch = 5000

epoch = 1

while True:
	losses = []
	usedPairs = []

	for e in range(samplesInEpoch):
		# Choosing an unused series pair
		while True:
			a = randint(0, len(S)-1)
			while True:
				b = randint(0, len(S)-1)
				if a != b: break
			if (a,b) not in usedPairs: break
		usedPairs.append((a, b))

		ida = S[a]
		idb = S[b]

		# Retrieving data about the two series
		aFound = bFound = False
		xa = xb = np.zeros((1, p))
		for i in range(S_data.shape[0]):
			if S_data[i, 0] == ida:
				xa = S_data[i, 1:]
				aFound = True
				if bFound:
					break
			if S_data[i, 0] == idb:
				xb = S_data[i, 1:]
				bFound = True
				if aFound:
					break
		if not aFound & bFound:
			print("Series were not found:", a, "-", b)

		# AlphaA, AlphaB, Loss, GammaDerivative, BetaDerivative <-- applyModelAndLoss
		Aa, Ab, L, DG, DB = applyModelAndLoss(a, b, xa, xb)
		print("Sample {}:".format(e + 1), "- Loss:", L)
		losses.append(L)

		# Updating the parameter matrices Gamma and Beta with derivatives
		Gamma = Gamma - learningRate * DG
		Beta = Beta - learningRate * DB

	notif.send("Epoch {} - Mean loss: {}".format(epoch, np.mean(losses)))
	epoch += 1
	ans = input("Should we go for another epoch? [y, n] ")
	if ans != "y":
		break

try:
	overwrite = True
	if os.path.exists(path_model):
		while True:
			ans = input("There is already a trained model file for these dimensions, should we overwrite it? [y, n]")
			if ans == "y":
				break
			elif ans == "n":
				overwrite = False
				break
	if overwrite:
		np.savez(path_model, Gamma=Gamma, Beta=Beta)
		print("Trained model saved at:", path_model)
except IOError as e:
	print("Error writing to file:", e)