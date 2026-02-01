import streamlit as st
from openai import OpenAI
import fitz
# PyMuPDF Function
def extract_text_from_pdf(uploaded_file):
    document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    document.close()
    return text
# Show title and description.
st.title("My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). ")
# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🔑")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt","pdf")
    )
    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )
    if uploaded_file and question:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o","gpt-4o-mini"]
        selected_model = st.selectbox("Select Model", model_options)
        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            stream=True,
        )
        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)