import pandas as pd 
import os
import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random as rd

base_path=('../Data/')
Animes=pd.read_csv(base_path+ 'AnimeList.csv',index_col='anime_id')

base_path2 = '../TreatedData/0_to_81'
path_users = base_path2 + '/users.pkl'
path_animes = base_path2 + '/animes.pkl'
path_ratings=base_path2 +'/ratings.pkl'

if os.path.exists(path_users):
	if os.path.exists(path_animes):
		if os.path.exists(path_ratings):
				with open(path_users, 'rb') as f:
					users = pickle.load(f)
				with open(path_animes, 'rb') as f:
					series = pickle.load(f)
				with open(path_ratings, 'rb') as f:
					rating_sparse = pickle.load(f)
ratings=rating_sparse.toarray()

Animes2=pd.read_csv(base_path+'/added_genre.csv',encoding = "ISO-8859-1")


neigh = NearestNeighbors( n_neighbors=10 )


Animes_genres=Animes2[Animes2.columns[32:]]

neigh.fit(Animes_genres)
def check(user_id,anime_id):
    if ratings[user_id,anime_id] == 0:
        return True
    else:
        return False

# Compute the Predicting Error For recommend system:
user_test_size = 200
#randomly choose 40 users to test
random_test_users = rd.sample(users,user_test_size)


#
#success =0
#failure =0
#for user_name in random_test_users:
#    # Fit the KNN model
#    user_test_id=users.index(user_name)
#    user_rating = list(ratings[user_test_id,:])
#    sorted_rating=sorted(range(len(user_rating)),key=lambda k:user_rating[k],reverse=True)
#    #find the 10 favorite animes
#    series_favorite = sorted_rating[:10]    
#    # find the position of similar_anime in ratings
#    success =0
#    failure =0
#    for i in range(10):
#        anime_id_favorite=series[series_favorite[i]]
#        if anime_id_favorite in Animes2['anime_id']:
#            # Find the similar animes
#            ser=Animes_genres.loc[Animes2['Id'][Animes2['anime_id'] == anime_id_favorite ]]
#            np2=ser.as_matrix()
#            similar_anime_id=list(neigh.kneighbors(np2.reshape(1,-1),return_distance= False)[0])
#            # Check if the similar_anime have been rated by users
#            for index in similar_anime_id:
#                animeid=Animes2['anime_id'][Animes2['Id'] == index ]
#                if  int(animeid) in series:
#                    ratings_anime_id= series.index(int(animeid))
#                    if user_rating[ratings_anime_id]!=0:
#                        if user_rating[ratings_anime_id] >= 7:
#                            success +=1
#                            print(1)
#                        if user_rating[ratings_anime_id] <5:
#                            failure +=1
#                            print(0)
#accurancy =failure/success
#print(accurancy)

    
#def coldrecommend(user_name):
#    coldlist=[0 for x in range(Animes_genres.shape[1])]
#    print('You are a new user,Please help me understand you ')
#    # get the preference vector of the user
#    for i in range(7):
#        print('If you like animes about below, Please choose the number')
#        print('If finished, input 7' )
#        for j in range(6):
#            print('{}. {}'.format(j,Animes_genres.columns[6*i+j]))
#
#        while(True):
#            answer=int(input())
#            if answer == 7:
#                break
#            else:
#                coldlist[6*i+answer]=1
#    nps=np.array(coldlist)
#    # Find similar animes based on the genres user like 
#    similar_anime_id=neigh.kneighbors(nps.reshape(1,-1),return_distance=False)
#    print ('We found some animes you may like')
#    print('Enjoy your animes')
#    for index in similar_anime_id:
#        anime_id = Animes2['anime_id'][index]
#        print(str(Animes['title_english'][anime_id]))
#
#print('Plsase enter your username')
##username = input()
#if username in users:
#    warmrecommend(username)
#else :
#    coldrecommend(username)
