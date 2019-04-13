import pickle, gzip, numpy
from matplotlib import pyplot as plt

from keras.layers import Input, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import keras.backend as K
from keras.optimizers import adam
from keras.utils import np_utils

def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)

def discriminator_model():

    # Model Type: Sequential, Provides one output defined by the classes
    # In our case we only care for True or False output (0,1)
    discriminator = Sequential()
    # Convolution Kernel to produce a tensor
    discriminator.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    # normalizes the matrix after convolution layer
    # each dimmension is kept in same scale
    discriminator.add(BatchNormalization(axis=-1))
    # max(0,x) - makes negative number equal zero
    # reduces training time and vanishing gradients
    discriminator.add(Activation('relu'))
    discriminator.add(Conv2D(32, (3, 3)))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(Activation('relu'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))

    discriminator.add(Conv2D(64, (3, 3)))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(Activation('relu'))
    discriminator.add(Conv2D(64, (3, 3)))
    discriminator.add(BatchNormalization(axis=-1))
    discriminator.add(Activation('relu'))
    # downsamples the input.
    # learns on the feautres. Reduces overfitting
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))

    # Need to flatten the output of the layers
    # so they can be input to dense layer
    discriminator.add(Flatten())

    # Dense layers are fully connected layers
    # serve for classification
    discriminator.add(Dense(512))
    discriminator.add(BatchNormalization())
    discriminator.add(Activation('relu'))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(1))

    # Provides a 0 or 1 output - F or T
    discriminator.add(Activation('sigmoid'))

    # for two classes we define a binary crossentropy loss
    # Adam improves SGD. User backpropagation to update weights
    discriminator.compile(loss='binary_crossentropy' , optimizer=adam_optimizer())

    return discriminator

def TestDiscriminator(X_train):
    # Generate noisy images
    noise = numpy.random.rand(210,28,28,1) # 210 noisy images
    K.cast(noise, dtype="float32")

    # Train on digits and noise
    X_train =  numpy.concatenate((noise[:200,:],X_train[0:1000,:]),axis=0)
    # Assign 0 for noise and 1 for digits
    trainLabels = numpy.append(numpy.zeros(200),numpy.ones(1000))

    # Begin Training!
    discriminator.fit(X_train, trainLabels, epochs=10)

    # Test model
    minitest = numpy.concatenate((noise[200:],X_test[0:32,:]),axis=0)
    results = discriminator.predict_classes(minitest)
    actual = numpy.concatenate((-1*numpy.ones(10),test[1][0:32]), axis=0)
    print("Predicted results (0/1): ", results.T)
    print("Actual Numbers (-1 = noise): ", actual)


if __name__== "__main__":

    # Load the dataset
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train, validation, test = pickle.load(f, encoding="latin1")

    X_train = train[0].reshape(train[0].shape[0], 28, 28, 1)
    X_validation = validation[0].reshape(validation[0].shape[0], 28, 28, 1)
    X_test = test[0].reshape(test[0].shape[0], 28, 28, 1)

    trainLabels = train[1]
    validationLabels = validation[1]
    testLabels = test[1]


    discriminator = discriminator_model()
    # discriminator.summary()
    TestDiscriminator(X_train)

    f.close()


