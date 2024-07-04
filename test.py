import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("new-digit-recognizer.keras")

# Function to preprocess the image


def preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, color_mode="grayscale",
                         target_size=(28, 28))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Reshape the image to add an extra dimension for batch size (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array = img_array / 255.0

    img.save('processed_image.png')
    return img_array


# Path to the new image
img_path = 'pic.png'  # Replace with your image path

# Preprocess the image
preprocessed_img = preprocess_image(img_path)
# hello world this is a test for copilot to see if it workd or not
# Predict the category of the image
predictions = model.predict(preprocessed_img)
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
print(f"The predicted class is: {predicted_class[0]}")
print(f"The predicted class is: {predicted_class}")
