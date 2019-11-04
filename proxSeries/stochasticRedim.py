#-*-coding:Utf-8 -*
"""Learning algorithm to produce a l-dim vector for each series, preserving distances between series
as much as possible, based on their ratings by all users."""

import numpy as np
import pickle as pkl
import os
from random import *
from collections import defaultdict
import matplotlib.pyplot as plt

# Reading and importing ratings matrix
print("Reading rating matrix R...")
import dataIO.ratingsRead
R = dataIO.ratingsRead.ratings.astype(float)
S = dataIO.ratingsRead.series
U = dataIO.ratingsRead.users

# Transforming R so that scores 1 to 10 range from -1 to 1, and 0 scores remain 0 (neutral)
if True:
	print("Preprocessing rating matrix R... 1/3")
	R = R - 1
	print("Preprocessing rating matrix R... 2/3")
	R = R + np.maximum(-np.sign(R), np.zeros(R.shape)) * 5.5
	print("Preprocessing rating matrix R... 3/3")
	R = R / 4.5 - 1

print("Imported rating matrix R:", R.shape)


def ratingDistance(itemA, pairs):
	"""Computes the rating distance between series at col itemA and at col itemB"""
	colA = R[:, itemA] * 1

	dsum = np.zeros((len(pairs), 1))

	for p in range(len(pairs)):
		colB = R[:, pairs[p]] * 1
		colB *= np.sign(colA)**2
		abso = np.abs(colB - colA * np.sign(colB)**2)
		n = np.count_nonzero(colB)
		if n != 0:
			dsum[p, 0] = sum(abso) / n

	return dsum


def outputDistance(alphaA, alphaB):
	"""Computes the L1-Norm between vectors alphaA and alphaB, normalizing by their dimension"""
	return sum(np.abs(alphaA - alphaB)) / l


# ------------------------------------------------------
#                 MODEL INITIALIZATION

l = 10
series = len(S)
Alpha = (np.random.rand(series, l) - 1) * 0.2
errors = []
sampleFreq = [0 for i in range(series)]
pairFreq = defaultdict(lambda: 0)

ans = input("Should we try to load previously trained model? [y, n] ")
path_model = "trainedModels/model_series_" + str(l) + ".npy"
path_model_info = "trainedModels/model_series_" + str(l) + "_info.pkl"

if ans == "y":
	if os.path.exists(path_model):
		try:
			Alpha = np.load(path_model)
			with open(path_model_info, "rb") as file:
				errors = pkl.load(file)
				sampleFreq = pkl.load(file)
				pairFreq = pkl.load(file)
			print("Loaded trained model for", l, "dimensions.")
		except (IOError, ValueError) as e:
			print("Error loading file:", e)
	else:
		print("There is no trained model for", l, "dimensions.")


# ------------------------------------------------------
#              PAIR-WISE STOCHASTIC LEARNING

sampleFreqHundred = [sum(sampleFreq[j] for j in range(i*100, min(i*100+100, len(sampleFreq)))) for i in range(series//100+1)]
plt.bar(range(series//100+1), sampleFreqHundred)

learningRate = 0.3
samplesPerEpoch = 10#50
pairsPerSample = 50#100

epoch = 0
run = 1
while True:
	error = 0
	for s in range(samplesPerEpoch):
		a = randint(0, series - 1)
		sampleFreq[a] += 1
		sampleFreqHundred[a//100] += 1

		plt.bar(range(series // 100 + 1), sampleFreqHundred)
		plt.pause(0.001)

		alphaA = Alpha[a, :]

		pairs = np.zeros((pairsPerSample, l))
		pairIds = []
		for p in range(pairsPerSample):
			while True:
				b = randint(0, series - 1)
				if b != a and b not in pairIds: break
			pairs[p, :] = Alpha[b, :]
			pairIds.append(b)
			pairFreq[b] += 1

		diff = pairs - alphaA
		dr = ratingDistance(a, pairIds)
		da = (np.sum(np.abs(diff), axis=1)/l).T.reshape(pairsPerSample, 1)
		sampleError = sum((dr - da)**2)[0] / pairsPerSample

		Alpha[a, :] = alphaA + learningRate * np.mean(diff * (1 - (dr / da)).reshape((dr.shape[0], 1)), axis=0)
		if run == 1:
			print("Sample:", s, "- Error:", sampleError, "- A:", a, "- mean dr:", np.mean(dr), "- mean da:", np.mean(da))
		error += sampleError

	errors.append(error / samplesPerEpoch)
	print("Epoch", epoch, "finished with mean error:", round(error / samplesPerEpoch, 4))
	epoch += 1
	run -= 1
	if run > 0:
		continue

	ans = input("Should we go for another epoch? [y, n, number-to-run] ")
	try:
		ans = int(ans)
		run = ans
	except ValueError:
		if ans != "y":
			break
		run = 1

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
		np.save(path_model, Alpha)
		with open(path_model_info, "wb+") as file:
			pkl.dump(errors, file)
			pkl.dump(sampleFreq, file)
			pkl.dump(pairFreq, file)
		print("Trained model saved at:", path_model)
except IOError as e:
	print("Error writing to file:", e)