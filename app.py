
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords




st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)
st.image("bgimg.jpg", width=120)


st.markdown("""
<h1 style='text-align: center; color: #ffffff;'>
💬 Chat Emotion Analyzer
</h1>
<p style='text-align: center; font-size:18px;'>
Analyze emotions from text instantly 🚀
</p>
""", unsafe_allow_html=True)
    
    
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, 
        #0000ff,   /* blue */
        #8000ff,   /* purple */
        #ff0000,   /* red */
        #ff8c00,   /* orange */
        #00ff7f    /* green */
    );
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    25% { background-position: 50% 100%; }
    50% { background-position: 100% 50%; }
    75% { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)  

###############################################################33
st.markdown("""
<style>

/* Full textarea container (glass layer) */
div[data-testid="stTextArea"] > div {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);

    border-radius: 15px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 8px !important;
}

/* 🔥 Remove Streamlit grey background layer */
div[data-testid="stTextArea"] > div > div {
    background: transparent !important;
}

/* Actual typing area */
div[data-testid="stTextArea"] textarea {
    background: transparent !important;
    color: white !important;
    border: none !important;
}

/* Placeholder */
div[data-testid="stTextArea"] textarea::placeholder {
    color: rgba(255,255,255,0.6);
}

/* Glow on focus */
div[data-testid="stTextArea"]:focus-within {
    border: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 0 0 25px rgba(255,255,255,0.25);
}

div[data-testid="stTextArea"] > div > div {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
# }
# </style>
# """, unsafe_allow_html=True) 

       
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
# your components here
st.markdown("</div>", unsafe_allow_html=True)


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
