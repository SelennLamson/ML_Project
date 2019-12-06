import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random as rd

base_path = '../TreatedData/0_to_81'
path_users = base_path + '/users.pkl'
path_animes = base_path + '/animes.pkl'
path_ratings=base_path +'/ratings.pkl'
animesnames=pd.read_csv('../Data/AnimeList.csv',index_col='anime_id')
# Open the user series and the ratings 
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

n_series=len(series)
n_users=len(users)
n_latent_factors = 100

# Do the martix factorization
user_input = keras.layers.Input(shape=(1,), name='user_input', dtype='int64')
user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
user_vec = keras.layers.Flatten(name='flat_user')(user_embedding)

series_input = keras.layers.Input(shape=(1,), name='series_input', dtype='int64')
series_embedding = keras.layers.Embedding(n_series, n_latent_factors, name='series_embedding')(series_input)
series_vec = keras.layers.Flatten(name='flat_series')(series_embedding)

sim = keras.layers.dot([user_vec, series_vec], name='dot-product', axes=1)
model = keras.models.Model([user_input, series_input], sim)

model.summary()
model.load_weights('../MatrixFactorization/KerasEmbeddingModel/KerasEmbeddingModel')

model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['mae'])
#print(model.layers[2].get_weights()[0].shape)
#print(model.layers[3].get_weights()[0].shape)
#get the matrix after factorization
ratings2=model.layers[2].get_weights()[0]

# Compute the Predicting Error For recommend system:
user_test_size = 40
#randomly choose 40 users to test
random_test_users = rd.sample(users,user_test_size)


ERRORLIST=[]
for users2 in random_test_users:
    ERROR = 0
    user_test_id = users.index(users2)
    users_cosine=cosine_similarity(ratings2[user_test_id].reshape(1,-1),ratings2)
    # find the most similar user
    sorted_similaruser=sorted(range(len(users_cosine.T)),key=lambda k:users_cosine.T[k],reverse=True)
    # find the user favorite anime
    user_rating = list(ratings[user_test_id,:])
    # check if the user is a old user which have given more than 10 animes ratings higher than 8
    checknewold=[i for i in user_rating if i>8]
    if len(checknewold)>10:
        sorted_rating=sorted(range(len(user_rating)),key=lambda k:user_rating[k],reverse=True)
        #test the 10 highest anime
        series_test = sorted_rating[:10]
        # Choose the rating 
        similaruser_rating=ratings[sorted_similaruser[1:200],:]
        #Choose the cosine similarity 
        similaruser_cosine=users_cosine.T[sorted_similaruser[1:200],:]
        i= 0
        for series in series_test:
            nozerorating=similaruser_rating[similaruser_rating[:,series]!=0,series]
            nozerousers_cosine=similaruser_cosine[similaruser_rating[:,series]!=0]
            if nozerousers_cosine.shape[0]>30:
                i+=1
                weight=nozerousers_cosine/sum(nozerousers_cosine)       
                # compute the predicting error
                predict_ratings=np.dot(weight.T,nozerorating)[0]    
                real_ratings=ratings[user_test_id][series]
                print('real score and predict_ratings are  {} and {}'.format(real_ratings,predict_ratings))
                ERROR += (real_ratings - predict_ratings ) ** 2 
        if ERROR != 0:
            ERRORLIST.append(ERROR/(i)) 
#Show the ERROR
m=[x for x in range(len(ERRORLIST))]
plt.plot(m,ERRORLIST)
plt.title("Compute the predict ERROR")
plt.yticks(np.linspace(0,10,11))
plt.show()