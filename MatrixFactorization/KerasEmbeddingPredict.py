import setup_tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
from dataIO.ratingsRead import read
import random
import matplotlib.pyplot as plt

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


# DATA PREPROCESSING (= training import)
ratings = ratings[:int(percent_of_users/100*n_users), :]
ratings_df = ratings.tocoo()
n_ratings = len(ratings_df.data)

ratings_std = ratings_df.data - np.mean(ratings_df.data)
ratings_std /= np.std(ratings_std)

# ratings_df = list(zip(ratings_df.row, ratings_df.col, ratings_std))
# random.shuffle(ratings_df)
# ratings_df = np.array(list(zip(*ratings_df))).T

ratings_df = np.array([ratings_df.row, ratings_df.col, ratings_std]).T

train = ratings_df[:int(split_test*n_ratings), :2]
train_labels = ratings_df[:int(split_test*n_ratings), 2]
n_train = train_labels.shape[0]

test = ratings_df[int(split_test*n_ratings):, :2]
test_labels = ratings_df[int(split_test*n_ratings):, 2]
n_test = test_labels.shape[0]
print(train.shape, test.shape)



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

train_predict = model.predict([train[:, 0], train[:, 1]], batch_size=2**20, verbose=1)
test_predict = model.predict([test[:, 0], test[:, 1]], batch_size=2**20, verbose=1)



margin = 0.5
train_neg = train_labels < - margin
train_pos = train_labels > margin
train_negr = train_predict.reshape(train_labels.shape) < - margin
train_posr = train_predict.reshape(train_labels.shape) > margin

sign_errors = (train_negr & ~train_neg) | (train_posr & ~train_pos)

print("Sign accuracy on training:", 100 - np.sum(sign_errors) / n_train * 100)


test_neg = test_labels < - margin
test_pos = test_labels > margin
test_negr = test_predict.reshape(test_labels.shape) < - margin
test_posr = test_predict.reshape(test_labels.shape) > margin

sign_errors = (test_negr & ~test_neg) | (test_posr & ~test_pos)

print("Sign accuracy on testing:", 100 - np.sum(sign_errors) / n_test * 100)





plot_percent = 100
plot_train = train_predict[:int(plot_percent/100*n_train)]
plot_test = test_predict[:int(plot_percent/100*n_test)]
plot_truth = train_labels[:int(plot_percent/100*n_train)]
plot_truth_test = test_labels[:int(plot_percent/100*n_test)]

plt.subplot(2, 1, 1)
plt.hist([plot_truth, plot_train.reshape(plot_truth.shape)], bins=10, range=(-3.71, 1.43), density=True, color=['g', 'b'], alpha=0.75)
plt.ylabel('distribution')
plt.xlabel('standardized rating')
plt.title('Train ratings distribution')
plt.legend(['truth', 'predicted'], loc='upper left')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.hist([plot_truth_test, plot_test.reshape(plot_truth_test.shape)], bins=10, range=(-3.71, 1.43), density=True, color=['g', 'b'], alpha=0.75)
plt.ylabel('distribution')
plt.xlabel('standardized rating')
plt.title('Test ratings distribution')
plt.legend(['truth', 'predicted'], loc='upper left')
plt.grid(True)

plt.show()


