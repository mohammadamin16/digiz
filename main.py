import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_train[2], cmap="gray")
plt.title(y_train[0])
plt.axis(False)
plt.show()
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1, ))
X_train = X_train / 255.
X_test = X_test / 255.
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
model = tf.keras.Sequential([
    layers.Conv2D(filters=10,
                  kernel_size=3,
                  activation="relu",
                  input_shape=(28,  28,  1)),
    layers.Conv2D(10,  3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(10,  3, activation="relu"),
    layers.Conv2D(10,  3, activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=1)
model.save("new-digit-recognizer.keras")
