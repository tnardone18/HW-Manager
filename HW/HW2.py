import streamlit as st
from openai import OpenAI
import anthropic
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

# LLM provider selection
llm_provider = st.sidebar.radio(
    "Choose LLM provider:",
    ["ChatGPT", "Claude"]
)

# Model selection
use_advanced = st.sidebar.checkbox("Use advanced model")

if llm_provider == "ChatGPT":
    model = "gpt-4o" if use_advanced else "gpt-4o-mini"
else:
    model = "claude-sonnet-4.5" if use_advanced else "claude-haiku-4.5"

st.sidebar.caption(f"Current model: {model}")

# Get API keys from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY")

# Check for required API key
if llm_provider == "ChatGPT" and not openai_api_key:
    st.warning("Please add your OpenAI API key to the Streamlit secrets.")
elif llm_provider == "Claude" and not anthropic_api_key:
    st.warning("Please add your Anthropic API key to the Streamlit secrets.")
else:
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
                
                prompt = f"{instruction}\n\nDocument: {document}"
                
                if llm_provider == "ChatGPT":
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        messages = [{"role": "user", "content": prompt}]
                        
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                        )
                        st.write_stream(stream)
                    except Exception as e:
                        st.error(f"OpenAI API error: {e}. Please check that your API key is valid.")
                    
                else:
                    try:
                        client = anthropic.Anthropic(api_key=anthropic_api_key)
                        
                        with st.empty():
                            response_text = ""
                            with client.messages.stream(
                                model=model,
                                max_tokens=1024,
                                messages=[{"role": "user", "content": prompt}]
                            ) as stream:
                                for text in stream.text_stream:
                                    response_text += text
                                    st.markdown(response_text)
                    except Exception as e:
                        st.error(f"Anthropic API error: {e}. Please check that your API key is valid.")