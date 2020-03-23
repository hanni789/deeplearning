import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import AE.convAE as aeConv

import matplotlib.pyplot as plt


(x_train, _), (x_test,_) = mnist.load_data()


img_size = x_train.shape[1]

print(x_train.shape)
x_train = np.reshape(x_train, [-1, img_size, img_size, 1]).astype('float32')
x_train /= 255
x_test = np.reshape(x_test, [-1, img_size, img_size, 1]).astype('float32')
x_test /= 255

# Adding noise 

noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)  
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)  
x_test_noisy = x_test + noise

ae = aeConv.ConvAE()
ae.train_model(data=x_train, bsize=10, epochs=4)

encoded_imgs = aeConv.Encoder(x_test)
predicted = ae.predict(x_test)

plt.figure(figsize=(40,4))
for i in range(10):
    # plt original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt encoder images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt predicted images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()