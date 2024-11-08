import streamlit as st
import requests
from gtts import gTTS
import os
import base64

# Hugging Face API key (replace with your key)
API_KEY = "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"

# Function to query Hugging Face models via API
def query_huggingface_model(model_name, data):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_name}",
        headers=headers, json=data
    )
    return response.json()

# Function for sentiment analysis and response generation
def virtual_psychologist(input_text):
    # Check for empty or invalid input
    if not input_text.strip():
        return "Please provide some input about how you're feeling.", "General Response", None

    # Step 1: Sentiment Analysis via Hugging Face Inference API
    sentiment_data = {"inputs": input_text}
    sentiment_result = query_huggingface_model("distilbert-base-uncased-finetuned-sst-2-english", sentiment_data)
    
    label = sentiment_result[0]['label']
    confidence = sentiment_result[0]['score']

    # Step 2: Display Sentiment Information
    sentiment_feedback = f"Your input sentiment is detected as **{label}** with confidence {confidence:.2f}.\n\n"

    # Step 3: Generate a Response Based on Sentiment
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

    # Step 4: Generate a Longer Response for the User using the Inference API
    text_gen_data = {"inputs": response}
    text_gen_result = query_huggingface_model("microsoft/DialoGPT-medium", text_gen_data)

    generated_text = text_gen_result[0]['generated_text']
    full_response = sentiment_feedback + generated_text

    # Convert response to speech using gTTS
    tts = gTTS(text=full_response, lang='en')
    tts.save("response.mp3")

    # Convert audio file to base64 for embedding in Streamlit
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    response_type = "Supportive Response" if "suicide" in input_text.lower() else "General Response"

    return full_response, response_type, audio_base64

# Streamlit App
st.title("Virtual Psychologist Assistant")
st.write("Share your feelings, and this assistant will analyze your sentiment and respond as a supportive psychologist.")

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Submit"):
    if user_input:
        full_response, response_type, audio_base64 = virtual_psychologist(user_input)

        # Display Text Responses
        st.write(f"Response Type: {response_type}")
        st.write(full_response)

        # Display Audio Response
        if audio_base64:
            audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.write("Please enter some text to analyze.")
