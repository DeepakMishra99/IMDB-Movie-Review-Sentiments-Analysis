import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st

# Load the IMDB DATASETS and word index
word_index = imdb.get_word_index()
reversed_word_index = {value : key for key, value in word_index.items()}

# Load the pre-trained model with RELU activation
model = load_model("simpleRNN_IMDB.h5")

# Step 2 : Helper Functions

# Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# Step 3 : Prediciton Function
def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

## Streamlit app

html_heading = """
<h3 style="color: #1c2833; background-color:#f4d03f;text-align: center;font-weight: bold;"><b>IMDb Movie Review Sentiment Analysis</b></h3>
"""

# Render the HTML in Streamlit
st.markdown(html_heading, unsafe_allow_html=True)

note = """
<h5 style="color: #e74c3c;"> Note : Enter a movie review to classify it as positive or negative.</h5>
"""

# Render the HTML in Streamlit
st.markdown(note, unsafe_allow_html=True)

# User Input

user_input = st.text_area('Movie Review')

if st.button('Classify'):

    sentiment,prediction_score=prediction_sentiment(user_input)

    #Display Result
    st.write(f'Sentiment:*{sentiment}*')
    st.write(f'Prediction_score: **{prediction_score}**')
else:
    st.write('Please enter the movie review.')
