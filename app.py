import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
responses = pickle.load(open('responses.pkl', 'rb'))

st.title("College FAQ Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    st.write("Bot:", responses[intent])
