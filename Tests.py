

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


def testar(ar):
	ar.append(50)


testlist = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
value = 800

print(testlist)
print(sorted_search(testlist, value, True))
testlist.insert(sorted_search(testlist, value, True), value)
print(testlist)

testar(testlist)
print(testlist)

import pickle
import numpy as np

# data = data.astype(np.int8)
# np.save('TreatedData/reduced.npy', data)

import math
import matplotlib.pyplot as plt

def prob(x, theta):
	return 2 * theta * x * math.exp(-theta * x * x)


test = "This is a string"

real_theta = 0.1
estimations = []
for n in range(1, 10):
	print(n)
	probas = [prob(x, real_theta) for x in range(1, n+1)]

	min_prob = min(probas)
	values = [i for i in range(1, n+1) for n in range(0, round(probas[i-1] // min_prob))]

	estimations.append(len(values) / sum([x**2 for x in values]))

plt.plot(estimations)
plt.show()