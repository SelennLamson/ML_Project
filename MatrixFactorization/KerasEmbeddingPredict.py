import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle

base_path = '../TreatedData/0_to_81'
path_users = base_path + '/users.pkl'
path_animes = base_path + '/animes.pkl'

users = []
series = []
if os.path.exists(path_users):
	if os.path.exists(path_animes):
		with open(path_users, 'rb') as f:
			users = pickle.load(f)
		with open(path_animes, 'rb') as f:
			series = pickle.load(f)

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
model.load_weights('KerasEmbeddingModel/KerasEmbeddingModel')

model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['mae'])
print(model.layers[2].get_weights()[0].shape)
print(model.layers[3].get_weights()[0].shape)

user_id = None
series_id = None
while True:
	ans = input("Username? ")
	if ans in users:
		user_id = users.index(ans)
		break
while True:
	ans = input("Anime ID? ")
	try:
		ans = int(ans)
		if ans in series:
			series_id = series.index(ans)
			break
	except ValueError:
		pass


pred_rating = model.predict([np.array(user_id).reshape(1, 1), np.array(series_id).reshape(1, 1)])

print(pred_rating)

