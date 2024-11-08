import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os

# Load sentiment analysis and text generation models
try:
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    text_gen_model = pipeline("text-generation", model="microsoft/DialoGPT-medium")
except Exception as e:
    st.error("Error loading models. Please check your environment or model configuration.")
    st.stop()

# Function for sentiment analysis and response generation
def virtual_psychologist(input_text):
    # Check for empty or invalid input
    if not input_text.strip():
        return "Please provide some input about how you're feeling.", "General Response", None

    try:
        # Step 1: Sentiment Analysis
        sentiment_result = sentiment_model(input_text)
        st.write("Sentiment result:", sentiment_result)  # Debugging output to check structure
        label = sentiment_result[0].get('label', 'NEUTRAL')
        confidence = sentiment_result[0].get('score', 0.5)

        # Step 2: Display Sentiment Information
        sentiment_feedback = f"Your input sentiment is detected as **{label}** with confidence {confidence:.2f}.\n\n"
    except Exception as e:
        st.error("Error during sentiment analysis. Please try again.")
        st.write(f"Details: {e}")
        return "Error analyzing sentiment. Please try again.", "General Response", None

    # Step 3: Generate a Response Based on Sentiment
    try:
        if confidence > 0.7:  # Threshold for confident sentiment analysis
            if label == 'POSITIVE':
                response = "I'm glad you're feeling positive! Tell me more about what’s bringing you joy, and let’s keep this energy up together."
            elif label == 'NEGATIVE':
                if "suicide" in input_text.lower() or "worthless" in input_text.lower():
                    response = ("I'm really sorry you're feeling this way, but please know you're not alone. "
                                "It's really important to talk to someone who can provide support. Would you like to share more "
                                "about what's been making you feel this way? You matter, and it's okay to reach out for help.")
                else:
                    response = "It sounds like you're going through a tough time. Want to share more about what’s on your mind? I'm here to help you navigate through it."
            else:
                response = "You seem to be feeling neutral. Do you have anything specific on your mind that you'd like to talk about?"
        else:
            response = "I'm not quite sure I understand. Could you elaborate a bit more? I'm here to listen."

        # Step 4: Generate a Longer Response for the User
        generated_text = text_gen_model(response, max_length=100, num_return_sequences=1)[0].get('generated_text', response)
        full_response = sentiment_feedback + generated_text
    except Exception as e:
        st.error("Error during response generation. Please try again.")
        st.write(f"Details: {e}")
        return "Error generating response. Please try again.", "General Response", None

    # Convert response to speech using gTTS
    try:
        tts = gTTS(text=full_response, lang='en')
        audio_path = "response.mp3"
        tts.save(audio_path)
    except Exception as e:
        st.error("Error generating audio. Please try again.")
        st.write(f"Details: {e}")
        audio_path = None

    return full_response, "Supportive Response" if "suicide" in input_text.lower() else "General Response", audio_path

# Streamlit App
st.title("Virtual Psychologist Assistant")
st.write("Share your feelings, and this assistant will analyze your sentiment and respond as a supportive psychologist.")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Submit"):
    if user_input:
        full_response, response_type, audio_file = virtual_psychologist(user_input)

        # Display Text Responses
        st.write(f"Response Type: {response_type}")
        st.write(full_response)

        # Display Audio Response if available
        if audio_file:
            st.audio(audio_file)
    else:
        st.write("Please enter some text to analyze.")
