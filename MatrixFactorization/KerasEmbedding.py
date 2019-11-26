import setup_tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataIO.ratingsRead import read
import random

users, series, ratings = read('../TreatedData/0_to_81')
n_series=len(series)
n_users=len(users)
n_latent_factors = 100

ratings_df = ratings.tocoo()
n_ratings = len(ratings_df.data)

ratings_df = list(zip(ratings_df.row, ratings_df.col, ratings_df.data))
random.shuffle(ratings_df)
ratings_df = np.array(list(zip(*ratings_df))).T

split = np.random.rand(n_ratings) < 0.7
train = ratings_df[split, :]
test = ratings_df[~split, :]
print(train.shape, test.shape)

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


batch_size=1024
epochs=1

history = model.fit([train[:, 0], train[:, 1]], train[:, 2], batch_size=batch_size,
					epochs=epochs, validation_data=([test[:, 0], test[:, 1]], test[:, 2]))

model.save_weights('KerasEmbeddingModel/KerasEmbeddingModel')

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