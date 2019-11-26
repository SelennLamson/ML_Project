import numpy as np
from dataIO.ratingsRead import read

users, series, ratings = read('../TreatedData/60_to_75')
n_series=len(series)
n_users=len(users)

ratings = ratings.toarray()

for i in range(10):
	r = np.zeros(ratings.shape)
	r[ratings == i + 1] = 1
	print(i+1, ':', np.sum(r))




# r = (np.random.random(4000000) - 0.5) * 6
# r = r**2
# print(np.mean(r))