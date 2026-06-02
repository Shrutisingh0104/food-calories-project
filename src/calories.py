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

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

top3 = np.argsort(prediction[0])[-3:][::-1]

st.subheader("Top Predictions")

for idx in top3:
    st.write(
        f"{class_names[idx]} : {prediction[0][idx]*100:.2f}%"
    )

print("Food:", food)
print("Calories:", calories[food], "kcal")