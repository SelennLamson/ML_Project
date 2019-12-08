import setup_tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
from dataIO.ratingsRead import read
import random
import matplotlib.pyplot as plt
import csv


base_path = '../TreatedData/0_to_81'
path_users = base_path + '/users.pkl'
path_animes = base_path + '/animes.pkl'

embedding_name = "Embedding_D200_RegL2"
standardized = True
n_latent_factors = 200
percent_of_users = 100
split_test = 0.8

random.seed(123456789)
np.random.seed(987654321)


users, series, ratings = read('../TreatedData/0_to_81')


n_series = len(series)
n_users = len(users)


# # DATA PREPROCESSING (= training import)
# ratings = ratings[:int(percent_of_users/100*n_users), :]
# ratings_df = ratings.tocoo()
# n_ratings = len(ratings_df.data)
#
# ratings_std = ratings_df.data - np.mean(ratings_df.data)
# ratings_std /= np.std(ratings_std)
#
# # ratings_df = list(zip(ratings_df.row, ratings_df.col, ratings_std))
# # random.shuffle(ratings_df)
# # ratings_df = np.array(list(zip(*ratings_df))).T
#
# ratings_df = np.array([ratings_df.row, ratings_df.col, ratings_std]).T
#
# train = ratings_df[:int(split_test*n_ratings), :2]
# train_labels = ratings_df[:int(split_test*n_ratings), 2]
# n_train = train_labels.shape[0]
#
# test = ratings_df[int(split_test*n_ratings):, :2]
# test_labels = ratings_df[int(split_test*n_ratings):, 2]
# n_test = test_labels.shape[0]
# print(train.shape, test.shape)



user_input = keras.layers.Input(shape=(1,), name='user_input', dtype='int64')
user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='user_embedding', embeddings_regularizer=keras.regularizers.l2(1e-6))(user_input)
user_vec = keras.layers.Flatten(name='flat_user')(user_embedding)

series_input = keras.layers.Input(shape=(1,), name='series_input', dtype='int64')
series_embedding = keras.layers.Embedding(n_series, n_latent_factors, name='series_embedding', embeddings_regularizer=keras.regularizers.l2(1e-6))(series_input)
series_vec = keras.layers.Flatten(name='flat_series')(series_embedding)

concat = keras.layers.Concatenate(axis=-1)([user_vec, series_vec])
sim = keras.layers.Dense(1, name='similarity', kernel_regularizer=keras.regularizers.l2(1e-6))(concat)

model = keras.models.Model([user_input, series_input], sim)

model.summary()
model.load_weights(embedding_name + '/' + embedding_name)

print(model.layers[2].get_weights()[0].shape)
print(model.layers[3].get_weights()[0].shape)

users_embedding_weights = model.layers[2].get_weights()[0]
series_embedding_weights = model.layers[3].get_weights()[0]

anime_data = []
with open("../Data/AnimeList.csv", "r", encoding="utf8") as csv_data:
	csv_reader = csv.reader(csv_data, delimiter=',')
	firstLine = True
	for row in csv_reader:
		if firstLine: firstLine = False
		else:
			anime_data.append(row)

anime_titles = dict()
for a in anime_data:
	anime_titles[int(a[0])] = a[1], int(float(a[15]))

print(anime_titles[15])

vecs = open("../TreatedData/animes_vecs.tsv", "w")
meta = open("../TreatedData/animes_meta.tsv", "w", encoding="utf8")
meta.write("Title\tScore\n")

for i in range(series_embedding_weights.shape[0]):
	vecs.write('\t'.join(map(str, list(series_embedding_weights[i, :]))) + '\n')
	if series[i] in anime_titles:
		meta.write(anime_titles[series[i]][0] + '\t' + str(anime_titles[series[i]][1]) + '\n')
	else:
		meta.write("unknown\t0.0\n")

vecs.close()
meta.close()


users_data = []
with open("../Data/UserList.csv", "r", encoding="utf8") as csv_data:
	csv_reader = csv.reader(csv_data, delimiter=',')
	firstLine = True
	for row in csv_reader:
		if firstLine: firstLine = False
		else:
			users_data.append(row)



users_meta = dict()
for a in users_data:
	gender = a[8]
	if a[8] == '':
		gender = 'nan'
	users_meta[a[0]] = gender, a[9]

vecs = open("../TreatedData/users_vecs.tsv", "w")
meta = open("../TreatedData/users_meta.tsv", "w", encoding="utf8")
meta.write("Username\tGender\tLocation\n")

for i in range(series_embedding_weights.shape[0]):
	vecs.write('\t'.join(map(str, list(users_embedding_weights[i, :]))) + '\n')
	username = users[i]
	if username in users_meta:
		meta.write(username + '\t' + users_meta[username][0] + '\t' + users_meta[username][1] + '\n')
	else:
		meta.write("nan\tnan\tnan\n")

vecs.close()
meta.close()