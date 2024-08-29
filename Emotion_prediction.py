import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model
st.write("""
# *Emotion Text Classification*

**This app predicts whether the text expresses emotions of Joy, Sadness, Anger, or Fear.**
""")

st.subheader("Enter your Text to know Emotion of it")
input_text = st.text_area("Enter your Text here")

if st.button("Submit"):
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    
    snowballstemmer = SnowballStemmer('english')
    def preprocess_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [snowballstemmer.stem(word) for word in text if word not in stop_words]
        return ' '.join(text)
    model = load_model('emotion_classification_rnn.h5')

    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    with open('label_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    
    input_text = preprocess_text(input_text)
    test_sequences = tokenizer.texts_to_sequences([input_text])
    maxlen = 35  # Ensure this matches the maxlen used during training
    X_test = pad_sequences(test_sequences, maxlen=maxlen)

   
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    predicted_emotion = encoder.inverse_transform(y_pred_labels)

    st.write(f"Your text has a Emotion of: {predicted_emotion[0]}")


