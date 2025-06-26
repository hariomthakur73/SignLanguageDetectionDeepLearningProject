from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import numpy as np
from PIL import Image

model = load_model("model.h5")

labels = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

st.title("Hand Gesture Recognition using Deep Learning")

img_input = st.camera_input("Take a photo of your hand")

if img_input:
    file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    input_img = gray.reshape(1, 256, 256, 1).astype("float32") / 255.0

    prediction = model.predict(input_img)
    predicted_label = labels[np.argmax(prediction)]

    st.image(img, caption="Captured Gesture", channels="BGR")
    st.success(f"Predicted Gesture: **{predicted_label}**")
