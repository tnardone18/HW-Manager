import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import tiktoken

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def count_tokens(messages, model="gpt-4o-mini"):
    """Count the total tokens in a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for message in messages:
        # every message has overhead tokens for role, content, etc.
        total_tokens += 4  # message overhead
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 2  # reply priming
    return total_tokens

# show title and description
st.title("My HW3 Question Answering Chatbot")

st.write("""
Welcome! This is a question answering chatbot powered by OpenAI's GPT models. Here's how it works:

- **Ask any question** and the chatbot will answer in simple, easy-to-understand language.
- **Provide up to 2 URLs** in the sidebar to give the chatbot extra context. It will read the webpage content and use it to answer your questions more accurately.
- **Choose your model** in the sidebar: "mini" (GPT-4o-mini, faster and cheaper) or "regular" (GPT-4o, more powerful).
- **Conversation memory**: This chatbot uses a **2,000 token buffer** to manage conversation history. The system prompt and URL context are always included and never discarded. For the rest of the conversation, the chatbot keeps as many recent messages as possible within a 2,000 token budget. Older messages are dropped first to stay within the limit. You can see the current token usage in the sidebar.
""")

openAI_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

# URL inputs in the sidebar
st.sidebar.markdown("### Provide URLs for Context")
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com")

# Load URL content when provided
url_context = ""
for i, url in enumerate([url1, url2], 1):
    if url.strip():
        content = read_url_content(url.strip())
        if content:
            # Truncate to avoid token limits
            truncated = content[:5000]
            url_context += f"\n\n--- Content from URL {i} ({url}) ---\n{truncated}"
            st.sidebar.success(f"URL {i} loaded successfully!")
        else:
            st.sidebar.error(f"URL {i} could not be loaded.")

# define the system prompt with URL context baked in
# this system prompt is NEVER discarded from the buffer
base_system_content = """You are a helpful question answering assistant. Your job is to get the user's question and answer it clearly and accurately.

IMPORTANT: Explain everything in simple terms that a 10 year old child can understand. Use easy words, short sentences, and fun examples when possible. Avoid jargon or complicated language.

After answering any question, always ask: "Do you want more info?"

If the user responds with "Yes", "yes", "yeah", "sure", "please", or any affirmative response:
- Provide additional details, examples, or related information about the previous topic
- Keep the explanation simple enough for a 10 year old
- Then ask again: "Do you want more info?"

If the user says "No", "no", "nope", or any negative response:
- Say something friendly like "Okay, no problem!"
- Then ask: "What else can I help you with?" to go back to the start"""

if url_context:
    base_system_content += f"""

You have been provided the following reference content from URLs. Use this content to help answer the user's questions when relevant. Always prioritize information from these URLs when the user's question relates to the content:
{url_context}"""

SYSTEM_PROMPT = {
    "role": "system",
    "content": base_system_content
}

# create an OpenAI client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

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
    
    # ---- TOKEN BUFFER STRATEGY (2,000 token limit) ----
    # The system prompt is NEVER discarded.
    # We start with the system prompt + the initial greeting,
    # then add recent messages from newest to oldest until
    # we hit the 2,000 token budget for conversation history.

    TOKEN_BUDGET = 2000

    # Always include the system prompt (never counted against the budget)
    buffered_messages = [SYSTEM_PROMPT]

    # Always include the initial assistant greeting
    initial_greeting = st.session_state.messages[0]
    buffered_messages.append(initial_greeting)

    # Calculate remaining token budget after system prompt and greeting
    base_tokens = count_tokens(buffered_messages, model_to_use)

    # Gather conversation messages (everything after the initial greeting)
    conversation_messages = st.session_state.messages[1:]

    # Add messages from most recent to oldest, staying within budget
    selected_messages = []
    running_tokens = base_tokens

    for msg in reversed(conversation_messages):
        msg_tokens = count_tokens([msg], model_to_use)
        if running_tokens + msg_tokens <= TOKEN_BUDGET + base_tokens:
            selected_messages.insert(0, msg)
            running_tokens += msg_tokens
        else:
            break  # stop adding older messages once budget is exceeded

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