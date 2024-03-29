import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import math

base_path = '../TreatedData/0_to_81'
path_users = base_path + '/users.pkl'
path_animes = base_path + '/animes.pkl'
path_ratings=base_path +'/ratings.pkl'
animesnames=pd.read_csv('../Data/AnimeList.csv',index_col='anime_id')
animesname=pd.read_csv('../Data/AnimeList.csv')
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

#get the rating matrix after factorization
ratings2=model.layers[2].get_weights()[0]
#define the constant number lambda
lamda=10
# Main UBCF recommendation algorithm
def recommend_rating(user_name,specify,popularity):
    # create the predicted animes list to store the predicted ratings
    predict_ratings=[0 for x in range(14480)]
    # get the position of users in rating matrix
    user_recommended = users.index(user_name)
    #compute the cosine similarity
    users_cosine=cosine_similarity(ratings2[user_recommended].reshape(1,-1),ratings2)
    print('Try to compute the similarity...')
    #Choose the anime haven't been rated and based on the popularity the user choose
    Anime_popularity=animesname['anime_id'][animesname['popularity']>int((1-popularity)*14478)]
    Anime_predicting=np.where(ratings[user_recommended,:]==0)[0]
    # sort the cosine similarity
    sorted_similaruser=sorted(range(len(users_cosine.T)),key=lambda k:users_cosine.T[k],reverse=True)
    print('Get the similar users ...')
    # Choose 10000 similar users 
    similarusers=10000
    # Choose the 10000 similar users rating records
    similaruser_rating=ratings[sorted_similaruser[1:similarusers],:]
    #Choose the cosine similarity 
    similaruser_cosine=users_cosine.T[sorted_similaruser[1:similarusers],:]
    # Compute the predicted rating by weighted average ratings
    for i in Anime_predicting:
        if series[i] in Anime_popularity:
            nozerorating=similaruser_rating[similaruser_rating[:,i]!=0,i]
            # Get the similar users' cosine similarity
            nozerousers_cosine=similaruser_cosine[similaruser_rating[:,i]!=0]
            # Adjust the similar user's cosine similarity by specify
            nozerousers_cosine = np.power(nozerousers_cosine,1+lamda*specify)
            if nozerousers_cosine.shape[0]>int(similarusers**0.5*5-20): 
                # predict the ratings by weighted average ratings
                weight=nozerousers_cosine/sum(nozerousers_cosine)    

                predict_ratings[i]=np.dot(weight.T,nozerorating)[0]
    # Sort the predicted ratings find the possible favorite anime
    sorted_predictrating=sorted(range(len(predict_ratings)),key=lambda k:predict_ratings[k],reverse=True)
    # print the recommended animes
    for favorite_anime in sorted_predictrating[:11]:
        favorite_animeid=series[favorite_anime]
        print('the score is {}'.format(predict_ratings[favorite_anime]))
        print('We recommend you to watch  {}'.format(animesnames['title_english'][favorite_animeid]))
    print('ENJOY YOUR ANIME')

print('Plsase enter your username')
username=input()
print('The rate you want to be recommended,from 1 to 10, 1 is popular,10 is specified ')
specify=int(input())
print('Please enter the popularity you can accept')
popularity = float(input())
print('Please wait, we are trying to recommend some animes')
recommend_rating(username,specify,popularity)

