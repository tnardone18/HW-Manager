import streamlit as st
from openai import OpenAI
import tiktoken
import sys
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_lab')
collection = chroma_client.get_or_create_collection('HW4Collection')                                        

def count_tokens(messages, model="gpt-4o-mini"):
    """Count the total tokens in a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for message in messages:
        total_tokens += 4
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 2
    return total_tokens

# show title and description
st.title("My HW4 Question Answering Chatbot")

st.write("""
Welcome! This is a question answering chatbot powered by OpenAI's GPT models. Here's how it works:

- **Ask any question** and the chatbot will answer in simple, easy-to-understand language.
- **Choose your model** in the sidebar: "mini" (GPT-4o-mini, faster and cheaper) or "regular" (GPT-4o, more powerful).
- **Conversation memory**: This chatbot uses a **2,000 token buffer** to manage conversation history. The system prompt is always included and never discarded. For the rest of the conversation, the chatbot keeps as many recent messages as possible within a 2,000 token budget. Older messages are dropped first to stay within the limit. You can see the current token usage in the sidebar.
""")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# define the system prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful question answering assistant. Your job is to get the user's question and answer it clearly and accurately.

IMPORTANT: Explain everything in simple terms that a 10 year old child can understand. Use easy words, short sentences, and fun examples when possible. Avoid jargon or complicated language.

After answering any question, always ask: "Do you want more info?"

If the user responds with "Yes", "yes", "yeah", "sure", "please", or any affirmative response:
- Provide additional details, examples, or related information about the previous topic
- Keep the explanation simple enough for a 10 year old
- Then ask again: "Do you want more info?"

If the user says "No", "no", "nope", or any negative response:
- Say something friendly like "Okay, no problem!"
- Then ask: "What else can I help you with?" to go back to the start"""
}

def add_to_collection(collection, text, file_name):
    client = st.session_state.client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            add_to_collection(collection, text, pdf_file.name)

# create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if collection.count() == 0:
    load_pdfs_to_collection('./su_orgs/', collection)

st.sidebar.write(f"Documents in ChromaDB: {collection.count()}")
  

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# display all messages in the chat history
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # --- RAG Retrieval: query ChromaDB for relevant context ---
    query_response = st.session_state.client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    rag_context = ""
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            source = results['ids'][0][i]
            rag_context += f"\n\n--- Retrieved from {source} ---\n{doc[:3000]}"

    # Inject RAG context into system prompt
    rag_system_prompt = dict(SYSTEM_PROMPT)
    if rag_context:
        rag_system_prompt["content"] += f"""

The following content was retrieved from course documents and may be relevant to the user's question. Use this information to answer accurately. If you use this information, let the user know it came from course materials.
{rag_context}"""

    TOKEN_BUDGET = 2000

    buffered_messages = [rag_system_prompt]

    initial_greeting = st.session_state.messages[0]
    buffered_messages.append(initial_greeting)

    base_tokens = count_tokens(buffered_messages, model_to_use)

    conversation_messages = st.session_state.messages[1:]

    selected_messages = []
    running_tokens = base_tokens

    for msg in reversed(conversation_messages):
        msg_tokens = count_tokens([msg], model_to_use)
        if running_tokens + msg_tokens <= TOKEN_BUDGET + base_tokens:
            selected_messages.insert(0, msg)
            running_tokens += msg_tokens
        else:
            break

    buffered_messages.extend(selected_messages)

    # Display token usage info in sidebar
    total_tokens = count_tokens(buffered_messages, model_to_use)
    st.sidebar.markdown("### Token Usage")
    st.sidebar.write(f"System prompt tokens: {count_tokens([SYSTEM_PROMPT], model_to_use)}")
    st.sidebar.write(f"Conversation buffer tokens: {running_tokens - base_tokens}")
    st.sidebar.write(f"Token budget for conversation: {TOKEN_BUDGET}")
    st.sidebar.write(f"Total messages in buffer: {len(buffered_messages)}")
    st.sidebar.write(f"Total messages in history: {len(st.session_state.messages)}")

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=buffered_messages,
        stream=True
    )
    
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})