

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
value = 850

print(testlist)
print(sorted_search(testlist, value, True))
testlist.insert(sorted_search(testlist, value, True), value)
print(testlist)

testar(testlist)
print(testlist)

import numpy as np

mat = np.zeros((5, 5))
vec = np.array([1, 2, 3, 4, 5])
vec = vec.reshape((5, 1))
print(vec.shape)
mat = mat - vec

print(mat)