import streamlit as st
from openai import OpenAI
import tiktoken
import sys
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_lab')
collection = chroma_client.get_or_create_collection('HW4Collection')                                        

# show title and description
st.title("My HW4 Question Answering Chatbot")

st.write("""
Welcome! This is a question answering chatbot powered by OpenAI's GPT models. Here's how it works:

- **Ask any question** and the chatbot will answer in simple, easy-to-understand language.
- **Choose your model** in the sidebar: "mini" (GPT-4o-mini, faster and cheaper) or "regular" (GPT-4o, more powerful).
- **Conversation memory**: This chatbot uses a **5-interaction buffer** to manage conversation history. The system prompt is always included and never discarded. The chatbot keeps the last 5 user-assistant exchanges. Older messages are dropped to stay within the limit.
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

def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    return soup.get_text()

# --- Chunking Method: Fixed-Size Character Chunking with Sliding Window Overlap ---
#
# Why fixed-size chunking?
#   Fixed-size chunking is the simplest and most predictable approach to splitting documents.
#   It segments text into equally sized pieces based on a character count. This ensures that
#   every chunk stays within the embedding model's input limits and produces consistently
#   sized vectors, which helps with fair similarity comparisons during retrieval.
#
# Why sliding window overlap?
#   When you split text at a fixed boundary, you risk cutting a sentence or idea in half.
#   By introducing an overlap between consecutive chunks, the end of one chunk and the
#   beginning of the next share some text. This maintains continuity of ideas across chunk
#   boundaries, so important context that falls near a split point is still captured in at
#   least one chunk. This improves retrieval quality because relevant information is less
#   likely to be lost at the edges.
#
# Parameters chosen:
#   - chunk_size=3000 characters: Large enough to capture meaningful context from each
#     HTML page, while small enough to stay within embedding model limits.
#   - overlap=500 characters: ~17% overlap preserves sentence continuity at boundaries
#     without creating too much redundancy.
#   - Each document produces exactly 2 chunks as required by the assignment. If the text
#     is shorter than one chunk, it is stored as a single chunk to avoid empty entries.

def chunk_text(text, chunk_size=3000, overlap=500):
    """
    Split text into fixed-size chunks with sliding window overlap.
    Each chunk is 'chunk_size' characters long, and consecutive chunks
    overlap by 'overlap' characters to preserve context at boundaries.
    Returns exactly 2 chunks per document (as required).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

        # Stop after 2 chunks as required by the assignment
        if len(chunks) >= 2:
            break

    return chunks

def add_to_collection(collection, text, doc_id):
    client = st.session_state.client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[doc_id],
        embeddings=[embedding]
    )

def load_htmls_to_collection(folder_path, collection):
    """
    Load HTML files, chunk each into 2 fixed-size pieces with overlap,
    and store each chunk as a separate document in ChromaDB.
    """
    folder = Path(folder_path)
    html_files = list(folder.glob("*.html"))
    for html_file in html_files:
        text = extract_text_from_html(html_file)
        if text.strip():
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    chunk_id = f"{html_file.name}_chunk{i}"
                    add_to_collection(collection, chunk, chunk_id)

# create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

# Only load HTML files into ChromaDB if the collection is empty (first run).
# Since we use PersistentClient, the data is saved to disk and persists across app restarts.
if collection.count() == 0:
    load_htmls_to_collection('./su_orgs/', collection)

html_file_count = len(list(Path('./su_orgs/').glob("*.html")))
st.sidebar.write(f"HTML files loaded: {html_file_count}")
st.sidebar.write(f"Chunks in ChromaDB: {collection.count()}")
  

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

    # --- 5-Interaction Conversation Buffer ---
    # An "interaction" = one user message + one assistant response (2 messages).
    # We keep the last 5 interactions (10 messages) plus the initial greeting.
    # The system prompt and RAG context are always included and never discarded.
    # Older interactions beyond the last 5 are dropped from the buffer.
    MAX_INTERACTIONS = 5
    MAX_MESSAGES = MAX_INTERACTIONS * 2  # 5 interactions = 10 messages (user + assistant pairs)

    # Start with system prompt (always included)
    buffered_messages = [rag_system_prompt]

    # Always include the initial assistant greeting
    initial_greeting = st.session_state.messages[0]
    buffered_messages.append(initial_greeting)

    # Get conversation messages (everything after the initial greeting)
    conversation_messages = st.session_state.messages[1:]

    # Keep only the last 5 interactions (last 10 messages)
    if len(conversation_messages) > MAX_MESSAGES:
        conversation_messages = conversation_messages[-MAX_MESSAGES:]

    buffered_messages.extend(conversation_messages)

    # Display buffer info in sidebar
    st.sidebar.markdown("### Conversation Buffer")
    st.sidebar.write(f"Max interactions stored: {MAX_INTERACTIONS}")
    st.sidebar.write(f"Messages in buffer: {len(conversation_messages)}")
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