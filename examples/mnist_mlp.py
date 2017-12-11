'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model
from examples import TensorResponseBoard

batch_size = 128
num_classes = 10
epochs = 20
# sample = 10000

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_test = x_test[:sample]
# y_test = y_test[:sample]
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))
log_dir = r'Z:\Master\Workspace\Python\Github\Keras_All\KerasWithComments\keras02\keras\output\examples\mnist_mlp'
tb = TensorResponseBoard.TensorResponseBoard(log_dir=log_dir,
                         histogram_freq=10,
                         batch_size=10,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=10,
                         embeddings_layer_names=embedding_layer_names,
                         embeddings_metadata='Z:\Master\Workspace\Python\Github\Keras_All\KerasWithComments\keras02\keras\output\examples\mnist_mlp\metadata.tsv',
                         val_size=len(x_test),
                         img_path='Z:\Master\Workspace\Python\Github\Keras_All\KerasWithComments\keras02\keras\output\examples\mnist_mlp\images.jpg',
                         img_size=[28, 28])

# tb = keras.callbacks.TensorBoard(log_dir='../output/examples/mnist_mlp',
#                                  histogram_freq=10,
#                                  batch_size=32,
#                                  write_graph=True,
#                                  write_grads=True,
#                                  write_images=True,
#                                  embeddings_freq=10,
#                                  embeddings_layer_names=embedding_layer_names,
#                                  embeddings_metadata=r"Z:\Master\Workspace\Python\Github\Keras_All\KerasWithComments\keras02\keras\output\examples\mnist_mlp\metadata.tsv")

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                     patience=3, verbose=1, mode='auto')
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
plot_model(model, to_file='../output/examples/mnist_mlp.png', show_shapes=True, show_layer_names=True)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tb, early_stop])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from PIL import Image
import numpy as np
import os
img_array = x_test.reshape(100, 100, 28, 28)
img_array_flat = np.concatenate([np.concatenate([x for x in row], axis=1) for row in img_array])
img = Image.fromarray(np.uint8(255 * (1. - img_array_flat)))
img.save(os.path.join(log_dir, 'images.jpg'))
np.savetxt(os.path.join(log_dir, 'metadata.tsv'), np.where(y_test)[1], fmt='%d')