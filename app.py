import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image, and the model will predict whether pneumonia is present.")



@st.cache_resource
def load_trained_model():
    return load_model("pneumonia_model.h5")


model = load_trained_model()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((150, 150))  
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict
    prediction = model.predict(img_array)
    predicted_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - \
        prediction[0][0]

    # Display result
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
