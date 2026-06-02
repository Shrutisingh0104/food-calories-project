import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("../food_mobilenetv2.h5")

# Classes must match training folders
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

st.title("🍔 Food Calorie Estimator using Deep Learning")

st.markdown("""
Upload a food image to:

✅ Identify the food item  
✅ Estimate calories  
✅ View prediction confidence  
""")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
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

        for idx in np.argsort(prediction[0])[::-1]:
            st.write(
                f"{class_names[idx]} : {prediction[0][idx] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

  