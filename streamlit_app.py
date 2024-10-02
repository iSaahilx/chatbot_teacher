import whisper
from openai import OpenAI
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import os
import base64
import re
# Load Whisper model
model = whisper.load_model("small")

# Initialize OpenAI API client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Title of the application
st.title("Chatbot-Language-Teacher")

# Language selection
language = st.selectbox(
    "Choose the language you want to learn:",
    [
        "Arabic", "English", "French", "German", "Hindi", "Italian",
        "Japanese", "Korean", "Malay", "Portuguese", "Russian",
        "Sanskrit", "Spanish", "Tamil", "Telugu", "Turkish", "Urdu"
    ]
)

# Mapping of language names to TTS language codes
language_code_map = {
    "Arabic": "ar", "English": "en", "French": "fr", "German": "de", "Hindi": "hi",
    "Italian": "it", "Japanese": "ja", "Korean": "ko", "Malay": "ms", 
    "Portuguese": "pt", "Russian": "ru", "Sanskrit": "hi", "Spanish": "es", 
    "Tamil": "ta", "Telugu": "te", "Turkish": "tr", "Urdu": "ur"
}



# Function to clean text and convert to speech
def clean_text_for_tts(text):
    # Replace or remove unwanted characters
    clean_text = text.replace('*', '')  # Remove asterisks
    clean_text = clean_text.replace('astrix', '')  # Remove unwanted phrases in French
    clean_text = clean_text.replace('tarankan', '')  # Remove unwanted phrases in Hindi
    
    # You can add more replacements as needed for other languages
    return clean_text

# Function to convert text to speech and play it
def text_to_speech(text, lang):
    # Clean the text before converting it to speech
    clean_text = clean_text_for_tts(text)
    
    # Generate TTS audio
    tts = gTTS(text=clean_text, lang=lang)
    tts.save("response.mp3")
    audio_file = open("response.mp3", "rb")
    audio_bytes = audio_file.read()
    audio_file.close()
    os.remove("response.mp3")
    st.audio(audio_bytes, format="audio/mp3")
   

# Voice recorder option
st.write("You can either record your voice or type a message to practice.")
recorded_audio = audio_recorder()

# If there's recorded audio, save and transcribe it
if recorded_audio:
    audio_file = "audio.mp3"
    with open(audio_file, "wb") as f:
        f.write(recorded_audio)
    
    # Transcribe the recorded audio using Whisper
    result = model.transcribe(audio_file, language)
    
    # Display the transcribed text in the chat as the user's message
    st.session_state.messages.append({"role": "user", "content": result["text"]})
    with st.chat_message("user"):
        st.markdown(result["text"])
else:
    # Input prompt for the user to type if no audio is recorded
    prompt = st.chat_input("Are you ready to practice?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize the chat messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Fetch the correct language code and pass it to the TTS function
            text_to_speech(message["content"], language_code_map[language])

# Generate response from OpenAI based on the conversation
if st.session_state.messages:
    with st.chat_message("assistant"):
        # Construct the conversation flow for OpenAI
        messages_for_api = [
            {
                "role": "system",
                "content": (
                    f"You are a professional linguistic teacher. And you teach {language} to your students who know English. I want you to simulate a real-time situation where your student might be conversing with someone in {language}. I would like you to have a conversation with your student where you help them learn the language. Also, give them hints to be able to translate or transliterate themselves. Don't give the answer yourself; instead, let me answer your conversation. Every time you ask a question, tell me the meaning of it in English too. Also, give a hint on how to answer that question (for example, if you as a seller ask, 'What do you want to buy?' give me a hint on how to respond. You could say, 'Think of how you would say (I want to buy... and then the item)' you could give the translations of the words that might be used in the response. After the conversation has ended, ask if they would like to practice again.")
            },
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]

        # Stream the assistant's response using the OpenAI API
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages_for_api,
            stream=True,
        )
        response = st.write_stream(stream)

    # Append assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Convert the assistant's response to speech using the language code
    text_to_speech(response, language_code_map[language])