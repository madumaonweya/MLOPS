import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


# Create the app title
st.title("Covid Image Classification App")

# instruction
st.write("Upload image of an X_Ray for prediction below")

# Create a file uploader
upload_file = st.file_uploader("Upload an image..", type=['jpg','jpeg', 'png'])

# check if the image is uploaded
if upload_file is not None: 
    # Open the image
    image = Image.open(upload_file)

    # display the image
    st.image(image, caption="Uploaded image")
    st.write("")

    # Preprocess the Image
    img = np.array(image)
    # resize the image
    img = tf.image.resize(img, (128,128))
    #normalize the image
    img = img/255.0
    # expand the image dimension
    img = np.expand_dims(img, axis = 0)
    #st.write(f'{img.shape}')

    # Load the train model
    model = load_model('vgg_model.h5')

    # Make predictions
    prediction = model.predict(img)

    # label
    class_labels = ["Covid", "Normal", "Pneumonia"]

    # Get the index that has the maximum value in the prediction array
    predicted_class_index = np.argmax(prediction)
    
    # Match the index with the corresponding label from class_labels list
    predicted_label = class_labels[predicted_class_index]

    # Display the predicted label
    st.write(f"Predicted label: {predicted_label}")
 

    #Display the prediction to the screen
    st.write(f" ### Predicted image: {predicted_label}")
