from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image










import pickle

#pickle_in = open('Malaria-Infected-Cells-Classification.h5', 'rb')
#model = pickle.load(pickle_in)
import tensorflow as tf
model = tf.keras.models.load_model('Malaria-Infected-Cells-Classification.h5')

import streamlit as st
st.write("""
         # Malaria-Infected-Cells-Classification
         """
         )
st.write("This is a simple image classification web app to classify a cell for malaria detection")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions


def import_and_predict(image_data, model):
    size = (140, 140)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(140, 140),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("Your Cell contains Malaria!")
    elif np.argmax(prediction) == 1:
        st.write("The cell does not contain Malaria!")
    else:
        st.write("Please upload new picture!")

    #st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)



