#-*-coding:Utf-8 -*
"""Launch this script to merge two rating imports with eachother."""

import numpy as np
import os
import pickle
import re


def sorted_search(ar, x, get_closest=False):#ar array x 
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

def get_coomatrix(A):
    rows=A.shape[0]
    columns=A.shape[1]
    row=[]
    column=[]
    value=[]
    for i in range(rows):
        for j in range(columns):
            if A[i][j]!=0:
                row.append(i)
                column.append(j)
                value.append(A[i][j])
    return row,column,value

base = '/Users/yimingwu/Documents/GitHub/ML_Project/TreatedData/'
folders = os.listdir(base)
folders = [f for f in folders if re.match(r'[0-9]+_to_[0-9]+', f)]

for i in range(len(folders)):
	print("[{}] - folder '{}'".format(i, folders[i]))

while True:
	f1 = input("Please select FIRST folder: ")
	try:
		f1 = int(f1)
		if 0 <= f1 < len(folders):
			break
	except ValueError:
		continue

while True:
	f2 = input("Please select SECOND folder: ")
	try:
		f2 = int(f2)
		if 0 <= f2 < len(folders) and f2 != f1:
			break
	except ValueError:
		continue

f1 = folders[f1]
f2 = folders[f2]

print("\nMerging folders", f1, "and", f2, "...")

val11 = int(f1.split('_')[0])
val12 = int(f1.split('_')[2])
val21 = int(f2.split('_')[0])
val22 = int(f2.split('_')[2])
valmin = min(val11, val21)
valmax = max(val21, val22)
joined = val11 <= val21 <= val12 or val21 <= val11 <= val22
output = (str(valmin) + '_to_' + str(valmax)) if joined else ('merged_' + f1 + '_and_' + f2)

print("Output folder will be:", output, "\n")

try:
    path_users = '/users.pkl'
    path_animes = '/animes.pkl'
    path_ratings = '/ratings.npy'
    path_info = '/info.pkl'

	# Reading folder n°1
    users1 = pickle.load(open(base + f1 + path_users, 'rb'))
    animes1 = pickle.load(open(base + f1 + path_animes, 'rb'))
    ratings1 = np.load(base + f1 + path_ratings)
	# Reading folder n°2
    users2 = pickle.load(open(base + f2 + path_users, 'rb'))
    animes2 = pickle.load(open(base + f2 + path_animes, 'rb'))
    ratings2 = np.load(base + f2 + path_ratings)
	# -----------------------------------------------------------------
	# ------------------------ TODO: CODE HERE ------------------------
	# -----------------------------------------------------------------
    # Merge the ratings2 to rating1
    userid,animeid,rating=get_coomatrix(ratings2)
    for every_no_zero_rating in range(len(userid)):
        fr=0
        fc=0
        if users2[userid[every_no_zero_rating]] in users1:
            fr=users1.index(users2[userid[every_no_zero_rating]])
            print(users2[userid[every_no_zero_rating]])
            print(users1[fr])
        else:
            users1.append(users2[userid[every_no_zero_rating]])
            ratings1=np.vstack((ratings1,np.zeros((1,ratings1.shape[1]),dtype=np.int8)))
            fr=ratings1.shape[0]-1
        if animeid[every_no_zero_rating] in animes1:
            fc=animes1.index(animeid[every_no_zero_rating])
        else:
            fc=sorted_insert(animes1,animeid[every_no_zero_rating])
            ratings1= np.hstack((ratings1[:, :fc], np.zeros((ratings1.shape[0], 1), dtype=np.int8), ratings1[:,fc:]))    
        percent=every_no_zero_rating/len(userid)
        if int(percent*10000)%10==0:
            print(percent*100)
        ratings1[fr][fc]=rating[every_no_zero_rating]        
#    For each non-zero value in ratings2 : r, c, v
#        username = users2[r]
#        animeid = animes2[c]
#        
#        fr = fc = 0
#        
#        if username in users1 --> r1:
#            fr = r1
#        else
#            users1.append(username)
#            ratings1 --> add row at the end
#            fr = last row index
#        
#        Using sorted_search and sorted_insert
#        if animeid in animes1 --> c1:
#            fc = c1
#        else
#            fc = sorted_insert(animes1, animeid)
#            ratings1 --> add column at the inserted index
#        
#        ratings[fr, fc] = v

	# Saving new data in output folder
    os.mkdir(base + output)
    pickle.dump(users1, open(base + output + path_users, 'wb'))
    pickle.dump(animes1, open(base + output + path_animes, 'wb'))
    np.save(base + output + path_ratings, ratings1)
    print("Data merged successfully. You can safely delete folders: '{}' and '{}'".format(f1, f2))
except IOError as e:
    print('Error with files:\n', e)


