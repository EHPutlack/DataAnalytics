import streamlit as st
from PIL import Image
img=Image.open('Logo.png')
st.image(img)
