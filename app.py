import streamlit as st
from transformers import pipeline

# Load sentiment analysis and text generation models
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
text_gen_model = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def virtual_psychologist(input_text):
    if not input_text.strip():
        return "Please provide some input about how you're feeling."

    sentiment = sentiment_model(input_text)[0]
    label = sentiment['label']
    confidence = sentiment['score']
    
    sentiment_feedback = f"Your input sentiment is detected as **{label}** with confidence {confidence:.2f}.\n\n"

    if confidence > 0.7:
        if label == 'POSITIVE':
            response = "I'm glad you're feeling positive!"
        elif label == 'NEGATIVE':
            response = "It sounds like you're going through a tough time."
        else:
            response = "You seem to be feeling neutral."
    else:
        response = "I'm not quite sure I understand. Could you elaborate a bit more?"

    generated_text = text_gen_model(response, max_length=100, num_return_sequences=1)[0]['generated_text']
    return sentiment_feedback + generated_text

# Streamlit App
st.title("Virtual Psychologist Assistant")
st.write("Share your feelings, and this assistant will analyze your sentiment and respond as a supportive psychologist.")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Submit"):
    if user_input:
        full_response = virtual_psychologist(user_input)
        st.write(full_response)
    else:
        st.write("Please enter some text to analyze.")
