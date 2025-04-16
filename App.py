import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('ai_or_not_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set up the Streamlit app layout
st.title("AI or NOT?")

# Add some text instructions
st.markdown("""
    This is a simple machine learning model that predicts if a given text is written by a **human** or **AI**.
    Just enter the text below and click 'Predict' to get the result.
""")

# Create a text input box for the user to enter an essay
user_input = st.text_area("Enter your text here:")

# When the button is pressed
if st.button('Predict'):
    # Vectorize the input text and make a prediction
    vectorized_input = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_input)
    
    # Display the result
    if prediction[0] == 1:
        st.write("**Prediction**: AI")
    else:
        st.write("**Prediction**: Human")
