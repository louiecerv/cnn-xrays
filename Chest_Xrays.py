#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import time

# Define the Streamlit app
def app():
    text = """Convolutional Neural Network Image Classifier Using Tensorflow and Keras on the Chest X-Rays Dataset"""
    st.subheader(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('chest-xray.jpg', caption='The Chest X-ray Dataset')

    text = """
    \nTensorFlow and Keras are a powerful combination for building 
    Convolutional Neural Networks (CNNs) for image classification tasks. 
    \nTensorFlow:
    TensorFlow is a powerful open-source library for numerical computation and 
    machine learning. It provides the core building blocks for creating 
    and training machine learning models.
    In CNNs for image classification, TensorFlow handles the low-level
    operations like:Tensor manipulation (tensors are multidimensional 
    arrays, the basic unit of data in TensorFlow)
    Mathematical computations
    Device optimization (utilizing GPUs for faster training)
    \nKeras:
    Keras is a high-level API built on top of TensorFlow, designed to 
    simplify model building and experimentation.It provides pre-built 
    layers and functions commonly used in deep learning models, 
    including CNNs. With Keras, you can define the architecture of your 
    CNN using a user-friendly interface. This includes specifying layers like:
    Convolutional layers (extract features from images)
    Pooling layers (reduce image dimensionality)
    Activation layers (introduce non-linearity)
    Fully connected layers (classify the extracted features)
    """
    st.write(text)

    with st.expander("How to use this App"):
         text = """Step 1. Go to Training page. Set the parameters of the CNN. Click the button to begin training.
         \nStep 2.  Go to Performance Testing page and click the button to load the image
         and get the model's output on the classification task.
         \nYou can return to the training page to try other combinations of parameters."""
         st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
