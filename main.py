import numpy as np
import tensorflow as ts
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index={values:keys for keys,values in word_index.items()}

model = load_model('simple_rnn_imdb.h5',compile=False)

## Helper function

# function to preprocess user input
def preprocess_text(text):
    words= text.lower().split()
    encoded_reviews=[word_index.get(word,2) + 3 for word in words]
    paded_review=sequence.pad_sequences([encoded_reviews],maxlen=500)
    return paded_review





# Streamlit App UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and classify it as **Positive** or **Negative**.")

user_input = st.text_area("✍️ Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid movie review.")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input, verbose=0)
        sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "😞 Negative"
        
        # Output result
        st.subheader("Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {prediction[0][0]}")
else:
    st.info("Enter a review and press 'Classify' to see the result.")