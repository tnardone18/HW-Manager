import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Show title and description
st.title("Document Summarizer")
st.write("Enter a URL below and get a summary!")

# URL input at the top of the screen
url = st.text_input("Enter a URL to summarize:")

# Sidebar options
st.sidebar.header("Summary Options")

# Summary type selection
summary_type = st.sidebar.radio(
    "Choose summary format:",
    [
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points"
    ]
)

# Language selection dropdown
output_language = st.sidebar.selectbox(
    "Choose output language:",
    ["English", "French", "Italian"]
)

# Model selection
use_advanced = st.sidebar.checkbox("Use advanced model")
model = "gpt-4o" if use_advanced else "gpt-4o-mini"
st.sidebar.caption(f"Current model: {model}")

# Get API key from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    if url:
        if st.button("Generate Summary"):
            # Read content from URL
            document = read_url_content(url)
            
            if document:
                # Build the prompt based on summary type
                if summary_type == "100 words":
                    instruction = f"Summarize the following document in exactly 100 words. Write the summary in {output_language}."
                elif summary_type == "2 connecting paragraphs":
                    instruction = f"Summarize the following document in 2 connecting paragraphs. Write the summary in {output_language}."
                else:
                    instruction = f"Summarize the following document in 5 bullet points. Write the summary in {output_language}."
                
                messages = [
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nDocument: {document}",
                    }
                ]

                # Generate summary using OpenAI API
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                st.write_stream(stream)
else:
    st.warning("Please add your OpenAI API key to the Streamlit secrets.")