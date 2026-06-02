import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_mobilenetv2.h5")

model = load_model()


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

st.title("🍔 Food Calorie Estimator")

uploaded_file = st.file_uploader(
    "Upload Food Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    try:
        
        img = image.resize((224, 224))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

       
        prediction = model.predict(img, verbose=0)

        predicted_index = np.argmax(prediction)

        food = class_names[predicted_index]
        confidence = float(np.max(prediction) * 100)

        st.success(f"🍽 Food: {food}")
        st.info(f"🔥 Estimated Calories: {calories[food]} kcal")
        st.write(f"Confidence: {confidence:.2f}%")

        
        st.subheader("Top Predictions")

        top_indices = np.argsort(prediction[0])[::-1]

        for idx in top_indices:
            st.write(
                f"{class_names[idx]} : {prediction[0][idx] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")