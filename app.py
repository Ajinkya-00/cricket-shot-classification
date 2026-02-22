import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cricket Shot Classification",
    page_icon="🏏",
    layout="centered"
)

# ---------------- DARK THEME + BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00ffcc;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}

.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🏏 Cricket Shot Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Based Shot Recognition System</div>', unsafe_allow_html=True)
st.write("")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cricket_shot_model.h5")
    return model

model = load_model()

# ---------------- LOAD CLASS INDICES ----------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index → class name)
CLASS_NAMES = {v: k for k, v in class_indices.items()}

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Cricket Shot Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    # Show Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing Shot..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class_name = CLASS_NAMES[predicted_class]

    # Display Results
    st.success(f"🎯 Predicted Shot: {predicted_class_name.upper()}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Probability Bar Chart
    st.subheader("Prediction Probability")

    prob_dict = {
        CLASS_NAMES[i]: float(prediction[0][i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & TensorFlow")