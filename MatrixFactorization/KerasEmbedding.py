import setup_tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataIO.ratingsRead import read
import random
import os
import pickle

# Hyper-parameters
embedding_name = "Embedding_D200_RegL2"
percent_of_users = 100
n_latent_factors = 200
split_test = 0.8

batch_size = 2**10
learning_rate = 1e-4
epochs = 1

users, series, ratings = read('../TreatedData/0_to_81')
n_series = len(series)
n_users = len(users)

random.seed(123456789)
np.random.seed(987654321)

ratings = ratings[:int(percent_of_users/100*n_users), :]
ratings_df = ratings.tocoo()
n_ratings = len(ratings_df.data)

ratings_std = ratings_df.data - np.mean(ratings_df.data)
ratings_std /= np.std(ratings_std)
# ratings_std = ratings_df.data / 10

ratings_df = list(zip(ratings_df.row, ratings_df.col, ratings_std))
random.shuffle(ratings_df)
ratings_df = np.array(list(zip(*ratings_df))).T

# split = np.random.rand(n_ratings) < split_test
train = ratings_df[:int(split_test*n_ratings), :]
test = ratings_df[int(split_test*n_ratings):, :]
print(train.shape, test.shape)

user_input = keras.layers.Input(shape=(1,), name='user_input', dtype='int64')
user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='user_embedding', embeddings_regularizer=keras.regularizers.l2(1e-5))(user_input)
user_vec = keras.layers.Flatten(name='flat_user')(user_embedding)

series_input = keras.layers.Input(shape=(1,), name='series_input', dtype='int64')
series_embedding = keras.layers.Embedding(n_series, n_latent_factors, name='series_embedding', embeddings_regularizer=keras.regularizers.l2(1e-5))(series_input)
series_vec = keras.layers.Flatten(name='flat_series')(series_embedding)

concat = keras.layers.Concatenate(axis=-1)([user_vec, series_vec])
sim = keras.layers.Dense(1, name='similarity', kernel_regularizer=keras.regularizers.l2(1e-5))(concat)
# sim = keras.layers.Dot([user_vec, series_vec], name='dot-product', axes=1)
# sim = keras.layers.Dot([user_vec, series_vec], normalize=True, name='cosine_sim', axes=1)

model = keras.models.Model([user_input, series_input], sim)

model.summary()

if os.path.exists(embedding_name):
	model.load_weights(embedding_name + '/' + embedding_name)
	print("Model loaded.")
else:
	os.mkdir(embedding_name)
	print("Model initialized.")

model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse', metrics=['mse', 'mae'])


print(train.shape, test.shape)

history = model.fit([train[:, 0], train[:, 1]], train[:, 2], batch_size=batch_size,
					epochs=epochs, validation_data=([test[:, 0], test[:, 1]], test[:, 2]))

model.save_weights(embedding_name + '/' + embedding_name)
print("Model saved.")

perfs = []
perf_file = embedding_name + '/' + embedding_name + '_perf.info'
if os.path.exists(perf_file):
	perfs = pickle.load(open(perf_file, 'rb'))

zipped = zip(history.history['loss'], history.history['val_loss'], history.history['mae'], history.history['val_mae'])
perfs += zipped
pickle.dump(perfs, open(perf_file, 'wb'))
print("Performance saved.")



train_data = train[:, :2]
train_outs = train[:, 2]
test_data = test[:, :2]
test_outs = test[:, 2]

train_results = model.predict([train_data[:, 0], train_data[:, 1]], batch_size=2**18)
test_results = model.predict([test_data[:, 0], test_data[:, 1]], batch_size=2**18)

train_accurate_signs = np.equal((train_outs > 0).reshape(train_results.shape), train_results > 0)
test_accurate_signs = np.equal((test_outs > 0).reshape(test_results.shape), test_results > 0)
train_acc = np.sum(train_accurate_signs) / (split_test * n_ratings)
test_acc = np.sum(test_accurate_signs) / ((1 - split_test) * n_ratings)
print("Train sign accuracy:", train_acc)
print("Test sign accuracy:", test_acc)


from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(history.history['loss'] , 'g')
plt.plot(history.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
