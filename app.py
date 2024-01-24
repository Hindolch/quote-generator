import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import  pad_sequences
import streamlit as st 
import numpy as np 

loaded_model = tf.keras.models.load_model("model.h5")
tokenizer = Tokenizer()

data = open('data2.txt', 'r').read()
for i in data.split(', '):
    tokenizer.fit_on_texts([i])






st.title('Quote generator')
st_text  = st.text_input('Enter a word..')
st_text_length = (st.text_input('Enter the length for the quote'))
st.write('You may need to adjust the length for a meaningful quote')

if st.button('generate'):
    for i in range(int(st_text_length)):
        token = tokenizer.texts_to_sequences([st_text])[0]
        pad = pad_sequences([token], maxlen=52, padding='pre')
        pred = np.argmax(loaded_model.predict(pad))
        for word, index in tokenizer.word_index.items():
            if index == pred:
                st_text = st_text + ' ' + word
st.success(st_text)