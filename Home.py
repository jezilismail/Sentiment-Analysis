import streamlit as st
from PIL import Image
import pickle 
import re
import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer


performance_report = pd.read_csv('assets/classification_report.csv')
######################################
def identity(x):
    return x

def clean_text(text):
  text = re.sub('(http|https):\/\/\S+', '', text)
  text = text.lower()
  text = re.sub(r'(.)\1+', r'\1\1', text)
  text = re.sub(r'[^A-Za-z ]', '', text)

  return text

stemmer = PorterStemmer()
def stem(text):
  text = [stemmer.stem(word) for word in text.split()]
  
  return text
######################################

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")
st.image("assets/cover.jpg", use_container_width=True)
st.markdown("""
    <h1 style='text-align: center; color: #03fcf0; font-family: Arial;'>
        Mental Health Analysis
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
This app predicts the emotion expressed in the text you provide.
We've trained a machine learning model on real, labeled text data that people shared online.

The model analyzes your input and returns the most likely emotional category based on learned patterns.
""")
text = st.text_area("Share your thoughts", height=200, placeholder='I feel amazing today!')
analyse = st.button('Analyse')

######################################
if analyse:
  if not text.strip(): 
    st.warning("Please enter some text for analysis.")
  else:
    model = pickle.load(open('assets/model.sav', 'rb'))
    encoder = pickle.load(open('assets/encoder.sav', 'rb'))
    vectorizer = pickle.load(open('assets/vectorizer.sav', 'rb'))

    cleaned = clean_text(text)
    stemmed = stem(cleaned)
    analysis = model.predict(vectorizer.transform([stemmed]))
    emotion = encoder.inverse_transform([analysis])[0]
    st.success(f"Detected Emotion: **{emotion}**")
######################################

st.markdown(
    "[Behind the scenes...](https://colab.research.google.com/drive/171q0z_nSf-aHA82c-8EHhQqmnvE4So1M#scrollTo=L6rE9HbTHMh8)"
)
# Initialize state only once
if "show_performance" not in st.session_state:
    st.session_state.show_performance = False

# Toggle button logic
def toggle_performance():
    st.session_state.show_performance = not st.session_state.show_performance

# Button with callback (no double-click issue)
st.button(
    "Hide Model Performance" if st.session_state.show_performance else "Show Model Performance",
    on_click=toggle_performance
)

# Conditional display
if st.session_state.show_performance:
    st.markdown('### Classification Report')
    st.dataframe(performance_report, use_container_width=True, hide_index=True)

    st.markdown('### Confusion Matrix')
    image = Image.open('assets/cnfsn_mtrx.png')
    st.image(image, caption='Confusion Matrix', use_container_width=True)