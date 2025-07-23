import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
import streamlit as st

# Initialize
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)  # NO punkt download needed

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("❌ Spam")
    else:
        st.header("✅ Not Spam")
