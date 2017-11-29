"""Trains a simple convnet on the FamilyFaces dataset

The dataset was derived from MS-Celeb-1M dataset.
Gets to 92.24% test accuracy after 50 epochs.
About 5 seconds per epoch on a Titan X Pascal GPU.
"""

from __future__ import print_function
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import cPickle as pkl
from sklearn.model_selection import train_test_split

# Define constants
batch_size = 6
num_classes = 6
epochs = 50
# Set random seed to get consistent result
random_seed = 1337
np.random.seed(random_seed)

# input image dimensions
img_rows, img_cols, img_channels = 224, 224, 3

# Load data from pickle file
with open('FamilyFaces.pkl', 'rb') as f:
    X = pkl.load(f)
    y = pkl.load(f)

# the data, shuffled and split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=random_seed)
print('X_train: ', X_train.shape)
print('X_test:', X_test.shape)

# Reshape tensor according to backend setting:
#   'th': 'channels_first'
#   'tf': otherwise
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

# Conver the original integer pixel value into float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Scale pixel value to range 0 to 1
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Save log file for TensorBoard visualization
tb_callback = keras.callbacks.TensorBoard(log_dir='./logs_cnn/batch' + str(batch_size) + '_epoch' + str(epochs), histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[tb_callback],
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save Model and Weights
model.save('familyfaces_cnn.h5')
