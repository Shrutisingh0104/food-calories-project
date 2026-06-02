import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("food_model.keras")

class_names = [
    "hamburger",
    "ice-cream",
    "pizza",
    "steak",
    "sushi"
]

calories = {
    "hamburger": 295,
    "ice-cream": 207,
    "pizza": 285,
    "steak": 679,
    "sushi": 200
}

img_path = input("Enter image path: ")

img = image.resize((224, 224))
img = np.array(img)
img = np.expand_dims(img, axis=0)

img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

prediction = model.predict(img, verbose=0)

prediction = model.predict(img_array)

food = class_names[np.argmax(prediction)]

print("Food:", food)
print("Calories:", calories[food], "kcal")