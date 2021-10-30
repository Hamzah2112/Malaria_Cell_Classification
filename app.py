
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf


def load_model():
  model = tf.keras.models.load_model('NEW_NEW.hdf5')
  return model

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction



image_1 = Image.open('aa.png')
image_2 = Image.open('download.jpg')
image_3 = Image.open('images.jpg')

with st.spinner('Model is being loaded..'):
  model=load_model()


st.image(image_1,use_column_width=True)

st.write("""
         ### Malaria Cell Classification Web Application
         """
         )



file = st.file_uploader("Upload the image of Cell", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    #prediction = np.argmax(predictions, axis=1)

    if prediction == 0:
        st.write("Alas! Your cell contains Malaria!")
        st.image(image_3,use_column_width=True)

    elif prediction ==1:
        st.write("Congratulations! The cell does not contain MALARIA")
        st.image(image_2,use_column_width=True)

    else:
        st.write('No, cell picture found! \n Please upload another picture ')





