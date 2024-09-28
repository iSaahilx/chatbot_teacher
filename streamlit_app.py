from openai import OpenAI
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import assemblyai as aai

aai.settings.api_key = "1c200dc8799a4f8da95d7dceb844ba8d"
transcriber = aai.Transcriber()


# transcript = transcriber.transcribe("./my-local-audio-file.wav")



st.title("Chatbot-Language-Teacher")


recorded_audio = audio_recorder()
if recorded_audio:
    audio_file = "audio.mp3"
    with open(audio_file, "wb") as f:
        f.write(recorded_audio)
        transcript = transcriber.transcribe("./audio.mp3")
        print(transcript.text)
        st.write(transcript.text)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Language selection
language = st.selectbox("Choose the language you want to learn:", [
    "Arabic",  "English", "French", "German", "Hindi","Italian", 
    "Japanese", "Korean", "Malay", "Portuguese", "Russian", 
    "Sanskrit", "Spanish", "Tamil", "Telugu", "Turkish","Urdu", 
])

# Display chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt for the user
if prompt := st.chat_input("Are you ready to practice?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create the chat completion with the new system message
        messages_for_api = [
            {"role": "system", "content": f"You are a professional linguistic teacher. And you teach {language} to your students who know English. I want you to simulate a real-time situation where your student might be conversing with someone in {language}. I would like you to have a conversation with your student where you help them learn the language. Also, give them hints to be able to translate or transliterate themselves. Don't give the answer yourself; instead, let me answer your conversation. Every time you ask a question, tell me the meaning of it in English too. Also, give a hint on how to answer that question (for example, if you as a seller ask, 'What do you want to buy?' give me a hint on how to respond. You could say, 'Think of how you would say (I want to buy... and then the item)' you could give the translations of the words that might be used in the response. After the conversation has ended, ask if they would like to practice again."},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        ]

        # Stream the response from OpenAI
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages_for_api,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Append the assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response})