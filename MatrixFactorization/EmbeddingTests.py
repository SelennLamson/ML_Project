import numpy as np
from dataIO.ratingsRead import read

users, series, ratings = read('../TreatedData/0_to_81')
print("Imported")
n_series=len(series)
n_users=len(users)

ratings = np.array(ratings.data)
print("Arrayed", ratings.shape)

ratings[ratings > 10] = 10
print(ratings[ratings > 10])
print("Min:", np.min(ratings), "Max:", np.max(ratings))

mean = np.mean(ratings)
std = np.std(ratings)
print("Mean:", mean)
print("Std:", np.std(ratings))

std_r = ratings - mean
std_r /= std

posr = std_r > 0
print("Percent positive:", np.sum(posr) / posr.shape[0])

print("Min:", np.min(std_r), "Max:", np.max(std_r))

fraction = 0
rem = std_r - std_r * fraction
print("Diffed")
mse = rem**2
mae = np.abs(rem)
print("Errored")

mse = np.mean(mse)
mae = np.mean(mae)

print("MSE:", mse)
print("MAE:", mae)





# r = (np.random.random(4000000) - 0.5) * 6
# r = r**2
# print(np.mean(r))