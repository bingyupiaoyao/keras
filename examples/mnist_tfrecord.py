'''MNIST dataset with TFRecords, the standard TensorFlow data format.

TFRecord is a data format supported throughout TensorFlow.
This example demonstrates how to load TFRecord data using
Input Tensors. Input Tensors differ from the normal Keras
workflow because instead of fitting to data loaded into a
a numpy array, data is supplied via a special tensor that
reads data from nodes that are wired directly into model
graph with the `Input(tensor=input_tensor)` parameter.

There are several advantages to using Input Tensors.
First, if a dataset is already in TFRecord format you
can load and train on that data directly in Keras.
Second, extended backend API capabilities such as TensorFlow
data augmentation is easy to integrate directly into your
Keras training scripts via input tensors.
Third, TensorFlow implements several data APIs for
TFRecords, some of which provide significantly faster
training performance than numpy arrays can provide because
they run via the C++ backend. Please note that this
example is tailored for brevity and clarity and not
to demonstrate performance or augmentation capabilities.

Input Tensors also have important disadvantages. In
particular, Input Tensors are fixed at model construction
because rewiring networks is not yet supported.
For this reason, changing the data input source means
model weights must be saved and the model rebuilt
from scratch to connect the new input data.
validation cannot currently be performed as training
progresses, and must be performed after training completes.
This example demonstrates how to train with input
tensors, save the model weights, and then evaluate the
model using the numpy based Keras API.

Gets to ~99.1% validation accuracy after 5 epochs
(high variance from run to run: 98.9-99.3).
'''
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras import layers

from tensorflow.contrib.learn.python.learn.datasets import mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def cnn_layers(x_train_input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x_train_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out

sess = K.get_session()

batch_size = 128
batch_shape = (batch_size, 28, 28, 1)
steps_per_epoch = 469
epochs = 5
num_classes = 10

# The capacity variable controls the maximum queue size
# allowed when prefetching data for training.
capacity = 10000

# min_after_dequeue is the minimum number elements in the queue
# after a dequeue, which ensures sufficient mixing of elements.
min_after_dequeue = 3000

# If `enqueue_many` is `False`, `tensors` is assumed to represent a
# single example.  An input tensor with shape `[x, y, z]` will be output
# as a tensor with shape `[batch_size, x, y, z]`.
#
# If `enqueue_many` is `True`, `tensors` is assumed to represent a
# batch of examples, where the first dimension is indexed by example,
# and all members of `tensors` should have the same size in the
# first dimension.  If an input tensor has shape `[*, x, y, z]`, the
# output will have shape `[batch_size, x, y, z]`.
enqueue_many = True

data = mnist.load_mnist()
x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[data.train.images, data.train.labels.astype(np.int32)],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    enqueue_many=enqueue_many,
    num_threads=8)

x_train_batch = tf.cast(x_train_batch, tf.float32)
x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

y_train_batch = tf.cast(y_train_batch, tf.int32)
y_train_batch = tf.one_hot(y_train_batch, num_classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

model_input = layers.Input(tensor=x_train_batch)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch])
train_model.summary()

# Fit the model using data from the TFRecord data tensors.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_model.fit(epochs=epochs,
                steps_per_epoch=steps_per_epoch)

# Save the model weights.
train_model.save_weights('saved_wt.h5')

# Clean up the TF session.
coord.request_stop()
coord.join(threads)
K.clear_session()

# Second Session to test loading trained model without tensors
x_test = np.reshape(data.validation.images, (data.validation.images.shape[0], 28, 28, 1))
y_test = data.validation.labels
x_test_inp = layers.Input(shape=(x_test.shape[1:]))
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test,
                                keras.utils.to_categorical(y_test),
                                num_classes)
print('\nTest accuracy: {0}'.format(acc))
