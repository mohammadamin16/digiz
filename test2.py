import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("new-model-new.keras")

# Function to preprocess the image


def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode="grayscale",
                         target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# Path to the new image
while True:

    img_path = input("Path: ")

# Preprocess the image
    preprocessed_img = preprocess_image(img_path)

# Predict the category of the image
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
    print(f"The predicted class is: {predicted_class[0]}")

# Optionally visualize the image and prediction
# plt.imshow(preprocessed_img.squeeze(), cmap="gray")
# plt.title(f"Predicted: {predicted_class[0]}")
# plt.axis(False)
# plt.show()
