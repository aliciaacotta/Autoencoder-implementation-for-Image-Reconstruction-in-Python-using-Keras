# -*- coding: utf-8 -*-
"""#Encoding and decoding using deep autoencoder (Shabd dataset - Kaggle)

single fully-connected neural layer as encoder and as decoder.

Create a deep autoencoder where the input image has a dimension of 1024.
Encode it to a dimension of 128 and then to 64 and then to 32.
Decode the 32 dimension image to 64 and then to 128 and finally reconstruct back to original dimension of 1024.
"""

from keras.layers import Input ,Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

"""Loading the Shabd dataset images. We want to reconstruct the images as output of the autoencoder.
"""

trainData= pd.read_csv("/content/shabd_train(grayscale).csv")
testData= pd.read_csv("/content/shabd_test(grayscale).csv")

print("Train Data Shape: ",trainData.shape)
print("Test Data Shape: ",testData.shape)

trainData.head()

testData.head()

Y_train=trainData["label"]
X_train=trainData.drop(labels=["label","Index"],axis=1)

print(X_train.shape)

Y_test=testData["label"]
X_test=testData.drop(labels=["label","Index"],axis=1)

print(X_test.shape)

"""Normalize all values between 0 and 1 and we will flatten the 32x32 images into vectors of size 1024.
"""

X_train=X_train/255.0
X_test=X_test/255.0

X_train=X_train.values.reshape(len(X_train),np.prod(X_train.shape[1:]))
X_test=X_test.values.reshape(len(X_test), np.prod(X_test.shape[1:]))
print(X_train.shape)
print(X_test.shape)

"""Convert the input image of dimensions 1024 to keras tensors"""

input_img= Input(shape=(1024,))
input_img

"""To build the autoencoder we will have to first encode the input image and add different encoded and decoded layer to build the deep autoencoder as shown below. The output layer needs to predict the probability of an output which needs to either 0 or 1 and hence we use sigmoid function.
For all the hidden layers for the encoder and decoder we use relu activation function for non-linearity.

"""

encoding_dim=32
# "encoded" is the encoded representation of the input
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=encoding_dim, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(encoded)

decoded = Dense(units=128, activation='relu')(decoded)
#reconstruct the image using sigmoid activation function
decoded = Dense(units=1024, activation='sigmoid')(decoded)

autoencoder=Model(input_img, decoded)

"""View the structure of deep autoencoder model"""

autoencoder.summary()

from keras.utils.vis_utils import plot_model
plot_model(autoencoder, show_shapes=True, show_layer_activations=True, show_dtype=True, expand_nested=True)

encoder = Model(input_img, encoded)

"""Structure of encoder model"""

encoder.summary()

from keras.utils.vis_utils import plot_model
plot_model(encoder, show_shapes=True, show_layer_activations=True, show_dtype=True, expand_nested=True)

# This is our encoded (32-dimensional) input
encoded_input = Input(shape=(32,))
# retrieve the last layer of the autoencoder model
num_decoder_layers = 3
decoder_layer = encoded_input
for i in range(-num_decoder_layers, 0):
    decoder_layer = autoencoder.layers[i](decoder_layer)

# create the decoder model
decoder = Model(encoded_input, decoder_layer)

"""Structure of decoder model"""

decoder.summary()

from keras.utils.vis_utils import plot_model
plot_model(decoder, show_shapes=True, show_layer_activations=True, show_dtype=True, expand_nested=True)

"""Compile the autoencoder model with adam optimizer uisng binary_crossentropy as the loss function. (pixels 0,1)
We use accuracy as the metrics used for the performance of the model.
"""

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""Train the autoencoder using the training data with 50 epochs and batch size of 256"""

autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

"""Predicting the test set using autoencoder to obtain the reconstructed image. Predict the test using the encoder to view the encoded images."""

encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

"""To view the original input, encoded images and the reconstructed images, we plot the images using matplotlib"""

plt.figure(figsize=(40, 4))
for i in range(10):

    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstructed images
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(predicted[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()
