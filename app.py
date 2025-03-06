import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os
import warnings
warnings.simplefilter("ignore", category=UserWarning)
# Load the model
filename = 'pokemon_classifier_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
type1_encoder = pickle.load(open('type1_encoder.pkl', 'rb'))

def predict_pokemon_type_production(image, loaded_model):
    try:
        img = Image.open(image).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = loaded_model.predict(img_array)
        predicted_type_index = np.argmax(prediction)
        predicted_type = "Unknown"
        
        if 'type1_encoder' in globals():
            predicted_type = type1_encoder.categories_[0][predicted_type_index]
        
        return predicted_type
    except Exception as e:
        return f"Error processing image: {e}"

# Streamlit UI
st.title("Pokémon Type Classifier")
st.write("Upload an image to predict the Pokémon type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save to a temporary file for processing
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make prediction
    prediction = predict_pokemon_type_production("temp_image.jpg", loaded_model)
    
    st.write(f"### Predicted Pokémon Type: {prediction}")
    
    # Clean up
    os.remove("temp_image.jpg")