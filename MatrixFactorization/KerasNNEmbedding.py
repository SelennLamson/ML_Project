import setup_tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataIO.ratingsRead import read

users, series, ratings = read('../TreatedData/60_to_75')
n_series=len(series)
n_users=len(users)
n_latent_factors = 64

ratings_df = ratings.tocoo()
ratings_df = (ratings_df.row, ratings_df.col, ratings_df.data)

split = np.random.rand(len(ratings_df[0])) < 0.5
train = (ratings_df[0][split], ratings_df[1][split], ratings_df[2][split])
test = (ratings_df[0][~split], ratings_df[1][~split], ratings_df[2][~split])

user_input = keras.layers.Input(shape=(1,), name='user_input', dtype='int64')
user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
user_vec = keras.layers.Flatten(name='flat_user')(user_embedding)
user_drop = keras.layers.Dropout(0.4)(user_vec)

series_input = keras.layers.Input(shape=(1,), name='series_input', dtype='int64')
series_embedding = keras.layers.Embedding(n_series, n_latent_factors, name='series_embedding')(series_input)
series_vec = keras.layers.Flatten(name='flat_series')(series_embedding)
series_drop = keras.layers.Dropout(0.4)(series_vec)

sim = keras.layers.dot([user_drop, series_drop], name='dot-product', axes=1)

nn_inp = keras.layers.Dense(96, activation='relu')(sim)
nn_inp = keras.layers.Dropout(0.4)(nn_inp)
# nn_inp=keras.layers.BatchNormalization()(nn_inp)
nn_inp = keras.layers.Dense(1, activation='relu')(nn_inp)

model = keras.models.Model([user_input, series_input],nn_inp)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='mse')

batch_size=512
epochs=10

history = model.fit([train[0], train[1]], train[2], batch_size=batch_size,
					epochs=epochs, validation_data=([test[0], test[1]], test[2]))


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