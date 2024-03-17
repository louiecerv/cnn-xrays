#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import time

# Define the Streamlit app
def app():


    text = """Convolutional Neural Network Image Classifier on the Chest X-Rays Dataset"""
    st.subheader(text)

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('smiles.jpg', caption='The Smiles Dataset')

    text = """
    \nDescribe this Streamlit App
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
