import numpy as np
import cPickle as pkl
from keras.models import load_model
from keras.utils import to_categorical


# Set random seed to get consistent result
random_seed = 1337
np.random.seed(random_seed)

num_classes = 6

with open('FamilyFaces.pkl', 'rb') as f:
    X = pkl.load(f)
    y = pkl.load(f)

# the data, shuffled and split between train and test sets
total_size = X.shape[0]
train_size = int(0.8 * total_size)
indices = np.random.permutation(total_size)
X = X[indices]
y = y[indices]
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print('X_train: ', X_train.shape)
print('X_test:', X_test.shape)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Load the saved model and weights
model = load_model('familyfaces_cnn.h5')
# model = load_model('familyfaces_vggface.h5')
# model = load_model('familyfaces_vggface_include_top.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
