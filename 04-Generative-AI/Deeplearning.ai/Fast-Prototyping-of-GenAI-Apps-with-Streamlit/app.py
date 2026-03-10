# import packages
from dotenv import load_dotenv
from groq import Groq
import os
import streamlit as st


# load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_data
def get_response(user_prompt, temperature):
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": user_prompt}  # Prompt
    ],
    temperature=temperature,  # A bit of creativity
    max_tokens=100  # Limit response length
)
    return response




st.title("Hello, GenAI!")
st.write("This is your first Streamlit app.")
user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

# Add a slider for temperature


temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,  
    help="Controls randomness: 0 = deterministic, 1 = very creative"
    )

with st.spinner("AI is working..."):
    response = get_response(user_prompt, temperature)    
    # print the response from Groq
    st.write(response.choices[0].message.content)





