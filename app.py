import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Emotion Analyzer", layout="centered")

# ------------------ CSS ------------------
st.markdown("""
<style>

/* Background gradient */
.stApp {
    background: linear-gradient(-45deg, #0000ff, #8000ff, #ff0000, #ff8c00, #00ff7f);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}

/* Gradient animation */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass card container */
.main-container {
    max-width: 600px;
    margin: auto;
    margin-top: 60px;
    padding: 30px;

    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);

    border-radius: 20px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}

/* Text */
h1, p, label {
    color: white !important;
    text-align: center;
}

/* Textarea glass */
div[data-testid="stTextArea"] > div {
    background: rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(15px);
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 8px !important;
}

div[data-testid="stTextArea"] > div > div {
    background: transparent !important;
}

div[data-testid="stTextArea"] textarea {
    background: transparent !important;
    color: white !important;
    border: none !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: rgba(255,255,255,0.6);
}

/* Button */
.stButton button {
    width: 100%;
    background: linear-gradient(45deg, #ff4b2b, #ff416c);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(255,75,43,0.6);
}

</style>
""", unsafe_allow_html=True)

# ------------------ UI START ------------------
#st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<h1>💬 Chat Emotion Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p>Analyze emotions from text instantly 🚀</p>", unsafe_allow_html=True)

# ------------------ NLP SETUP ------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ------------------ INPUT ------------------
user_input = st.text_area("Enter your message here:", placeholder="I am feeling very happy today")

# ------------------ BUTTON ------------------
if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]

        # Emotion colors
        emotion_colors = {
            "happy": "#00ff9f",
            "sad": "#4da6ff",
            "angry": "#ff4d4d",
            "fear": "#b366ff",
            "love": "#ff66cc"
        }

        color = emotion_colors.get(prediction.lower(), "#ffffff")

        # Chat-style UI
        st.markdown(f"""
        <div style="
            background: rgba(0,0,0,0.3);
            padding: 12px;
            border-radius: 12px;
            margin-top: 15px;
        ">
            <b>You:</b> {user_input}
        </div>

        <div style="
            background: rgba(255,255,255,0.1);
            padding: 12px;
            border-radius: 12px;
            margin-top: 10px;
            border-left: 5px solid {color};
            box-shadow: 0 0 15px {color};
        ">
            <b>AI:</b> Emotion → {prediction}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
