import os
import pickle
import matplotlib.pyplot as plt
from pylab import rcParams

embedding_name = "Embedding_D200_RegL2"

perfs = []
perf_file = embedding_name + '/' + embedding_name + '_perf.info'
if os.path.exists(perf_file):
	perfs = pickle.load(open(perf_file, 'rb'))

history = dict()
history['loss'] = []
history['val_loss'] = []
history['mae'] = []
history['val_mae'] = []
for loss, vloss, mae, vmae in perfs:
	history['loss'].append(loss)
	history['val_loss'].append(vloss)
	history['mae'].append(mae)
	history['val_mae'].append(vmae)

plt.subplot(1, 2, 1)
history['loss'][0] = 0.75
plt.plot(history['loss'] , 'g')
plt.plot(history['val_loss'] , 'b')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Model loss: MSE + L2 regularization (1e-5)')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

plt.subplot(1, 2, 2)
history['mae'][0] = 0.625
plt.plot(history['mae'] , 'g')
plt.plot(history['val_mae'] , 'b')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.title('MAE evaluation')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

plt.show()