import pandas as pd 
import os
import pickle
import random
import numpy as np

from sklearn.neighbors import NearestNeighbors

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

def setcatgory(u):
    for anime in range(14478):
        genrestr = str(Animes['genre'][anime])
        nospace = genrestr.replace(' ','')
        genrelist = nospace.split(',') # split by ,
        if anime == 9000:
            print('70%')
        if anime==11000:
            print('80%')
        for genre in genrelist:
            
            if genre in Animes.columns:
                Animes[genre][anime]=1
            else:
                zerolist = [0 for x in range(14478)]
                columns = len(Animes.columns)
                Animes.insert(columns,genre,zerolist)
                Animes[genre][anime]=1



Animes2=pd.read_csv(base_path+'/added_genre.csv',encoding = "ISO-8859-1")


neigh = NearestNeighbors( n_neighbors=10 )


Animes_genres=Animes2[Animes2.columns[32:]]
neigh.fit(Animes_genres)
def check(user_id,anime_id):
    if ratings[user_id,anime_id] == 0:
        return True
    else:
        return False

def warmrecommend(user_name):
    # Fit the KNN model
    print('WELCOME BACK')
    user_test_id=users.index(user_name)
    user_rating = list(ratings[user_test_id,:])
    sorted_rating=sorted(range(len(user_rating)),key=lambda k:user_rating[k],reverse=True)
    #find the 10 favorite animes
    series_favorite = sorted_rating[:10]    
    # Check if the similar_anime have been rated by users
    # find the position of similar_anime in ratings
    recommend_title=[]
    for i in range(10):
        anime_id_favorite=series[series_favorite[i]]
        # Find the similar animes
        ser=Animes_genres.loc[Animes2['Id'][Animes2['anime_id'] == anime_id_favorite ]]
        np=ser.as_matrix()
        similar_anime_id=list(neigh.kneighbors(np.reshape(1,-1),return_distance= False)[0])
        # Check if the similar_anime have been rated by users
        for index in similar_anime_id:
            animeid=Animes2['anime_id'][Animes2['Id'] == index ]
            if  int(animeid)  not in series:
                recommend_title.append(str(Animes['title_japanese'][int(animeid)]))
            else :
                anime_id_check=series.index(int(animeid))
                if check(user_test_id,anime_id_check):
                    recommend_title.append(str(Animes['title_japanese'][int(animeid)]))
     #randomly recommend 10 anime users may like  
    randomlist= random.sample(range(len(recommend_title)),10)
    for k in randomlist:
        print('WE Find some similar animes you like \t  {} '.format(recommend_title[int(k)]))
        
    
def coldrecommend(user_name):
    coldlist=[0 for x in range(Animes_genres.shape[1])]
    print('You are a new user,Please help me understand you ')
    # get the preference vector of the user
    for i in range(7):
        print('If you like animes about below, Please choose the number')
        print('If finished, input 7' )
        for j in range(6):
            print('{}. {}'.format(j,Animes_genres.columns[6*i+j]))

        while(True):
            answer=int(input())
            if answer == 7:
                break
            else:
                coldlist[6*i+answer]=1
    nps=np.array(coldlist)
    # Find similar animes based on the genres user like 
    similar_anime_id=neigh.kneighbors(nps.reshape(1,-1),return_distance=False)
    print ('We found some animes you may like')
    print('Enjoy your animes')
    for index in similar_anime_id:
        print(str(Animes2['title'][index]))

print('Plsase enter your username')
username = input()
if username in users:
    warmrecommend(username)
else :
    coldrecommend(username)


        
   