import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if they aren't already on the system
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 1. Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 2. Define the exact same cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split() 
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# 3. Build the Streamlit User Interface
st.title("💬 Chat Emotion Analyzer")
st.write("Type a message below to detect the emotion behind the text!")

# Create a text input area
user_input = st.text_area("Enter your message here:", placeholder="I am feeling very happy today")

# Create a button to trigger the prediction
if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Clean the input
        cleaned_input = clean_text(user_input)
        
        # Vectorize the input
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Predict the emotion
        prediction = model.predict(vectorized_input)[0]
        
        # Display the output
        st.success(f"**Output: Emotion - {prediction}**")