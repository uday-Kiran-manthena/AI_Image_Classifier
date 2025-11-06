import streamlit as st 
import numpy as np 
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from scipy.ndimage import shift 

model_file = 'models/best_model.h5'
digit_lables = [str(i) for i in range(10)]

@st.cache_resource
def get_model():
    model = load_model(model_file)
    return model

def format_and_center(image: Image.Image) -> np.ndarray:
    gray = image.convert('L')
    inverted = ImageOps.invert(gray)    
    resized = inverted.resize((28, 28))
    pixel_array = np.array(resized, dtype=np.float32) / 255.0
    
    def center_digit(arr):
        coords = np.argwhere(arr > 0)
        if coords.size == 0:
            return arr
        cy, cx = coords.mean(axis=0)
        rows, cols = arr.shape
        shift_x = int(np.round(cols / 2 - cx))
        shift_y = int(np.round(rows / 2 - cy))
        return shift(arr, [shift_y, shift_x], mode='constant')
    centered = center_digit(pixel_array)

    final_input = np.expand_dims(centered, axis=(0, -1))
    return final_input

st.title("AI Image Classifier")
st.write("Upload an image of digit (0-9) and let the model predict it!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"]) 

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=200)
    input_data = format_and_center(image)

    model = get_model()
    prediction = model.predict(input_data)[0]

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100 

    st.markdown(f"### Predicted Digit: {digit_lables[predicted_class]}")
    st.markdown(f"### Confidence: {confidence:.2f}%")

    st.write("### Prediction Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"Digit {i}: {prob*100:.2f}%")