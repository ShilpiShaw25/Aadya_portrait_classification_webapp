
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle

from keras.layers import GlobalAveragePooling2D
from keras.models import Model


IMG_SIZE = (224, 224)
IMAGE_NAME = "potrait.png"

PREDICTION_LABELS = [
    "Tom Holland","James Bond","Taylor Swift","Emma Watson","Scarlett Johanson" ]
PREDICTION_LABELS.sort()


# functions
@st.cache_resource
def get_mobilenetV2_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions

# get the featurization model
featurized_model = get_mobilenetV2_model()
# load Cataract model
Prediction_model = load_sklearn_models("best_model_mlp")



#Building the website

#title of the web page
st.title("Portrait Classification Webapp")

#setting the main picture
#st.image( "https://mediniz-images-2018-100.s3.ap-south-1.amazonaws.com/post-images/chokhm_1663869443.png",caption = "ABC")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.write("This web app predicts the name of the celebrity based on the sketch provided")



#setting file uploader
#you can change the label name as your preference
image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")
if image:
  #validating the image type
  image_type = image.type.split("/")[-1]
  if image_type not in ['jpg','jpeg','png','jfif']:
      st.error("Invalid file type : {}".format(image.type), icon="🚨")
  else:
      #displaying the image
      st.image(image, caption = "User Uploaded Image")


      if image:
        user_image = Image.open(image)
        # save the image to set the path
        user_image.save(IMAGE_NAME)

        #get the features
        with st.spinner("Processing......."):
          image_features = featurization(IMAGE_NAME, featurized_model)
          model_predict = Prediction_model.predict(image_features)
          model_predict_proba = Prediction_model.predict_proba(image_features)
          probability = model_predict_proba[0][model_predict[0]]
          col1, col2 = st.columns(2)

          with col1:
              st.header("Celebrity Name")
              st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
          with col2:
              st.header("Prediction Probability")
              st.subheader("{}".format(probability))



