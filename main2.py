import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"We have {len(X_train)} images in the training set and {len(X_test)} images in the test set.")

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Visualize the first training image
plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title(y_train[0])
plt.axis(False)
plt.show()

# Build the model
model = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3,
                  activation="relu", input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("digit_recognizer_model.h5")
