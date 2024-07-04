import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255


# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_images)

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with a learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 2
batch_size = 64

# Train the model with data augmentation
model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
          epochs=epochs,
          validation_data=(test_images, test_labels),
          steps_per_epoch=len(train_images) // batch_size)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Function to preprocess and predict custom images


def preprocess_and_predict(image_path, model):
    from PIL import Image
    import numpy as np

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).astype('float32') / 255  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape

    # Predict
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label[0]


# Example usage
while (True):
    custom_image_path = input("Enter image address: ")
    predicted_label = preprocess_and_predict(custom_image_path, model)
    print(f"Predicted label: {predicted_label}")
