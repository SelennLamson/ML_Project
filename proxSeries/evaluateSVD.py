#-*-coding:Utf-8 -*
"""Computes the SVD of rating matrix R with scipy.sparse.linalg.svds, which is memory-constant."""

import numpy as np
import os

# Reading and importing ratings matrix
print("Reading rating matrix R...")
import dataIO.ratingsRead
R = dataIO.ratingsRead.ratings.astype(float)

k = 5000
path_svd = "trainedModels/svd_" + str(k) + ".npz"
if os.path.exists(path_svd):
	loaded = np.load(path_svd)
	U = loaded['U']
	S = loaded['S']
	Vt = loaded['Vt']

	print("Reconstructing R from:")

	print("U:", U.shape)
	print("S:", S.shape)
	print("Vt:", Vt.shape)

	while True:
		ans = input("SVD is available in " + str(k) + " dimensions. How much sub-SVDs should we compute error for? ")
		try:
			ans = int(ans)
			if 0 < ans < 1000:
				break
			else:
				print("Should be between 1 and 999.")
		except ValueError:
			continue

	for i in range(ans):
		newk = int(min(k / ans * (i+1), k))

		R_rec = U[:, -newk:] @ np.diag(S[-newk:]) @ Vt[-newk:, :]
		R_rec = np.round(R_rec)
		diff = R - R_rec
		abso = np.abs(diff)
		mean_error = np.mean(abso)
		max_error = np.max(abso)
		norm = np.linalg.norm(diff) / np.linalg.norm(R)
		print("Error for k =", newk, ":", round(norm * 100, 2), "%")
