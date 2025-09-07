import streamlit as st
import requests
import re
import time
import hashlib
from datetime import datetime

# Configuration

BACKEND_URL = "http://127.0.0.1:8000"
ASK_ENDPOINT = f"{BACKEND_URL}/ask"
FEEDBACK_ENDPOINT = f"{BACKEND_URL}/feedback"

# Debug Backend Health Check

def test_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"Health check: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
test_backend()

# Page Config

st.set_page_config(
    page_title="Visit Ethiopia Travel Assistant",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Session State Initialization

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False
if "waiting" not in st.session_state:
    st.session_state.waiting = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  

# Custom CSS - Grok-Inspired with Theme Toggle

def get_theme_css():
    if st.session_state.theme == "dark":
        return """
        :root {
            --bg: #1C2526; /* Grok dark background */
            --text: #FFFFFF; /* White text */
            --accent: #1DA1F2; /* Grok blue */
            --secondary: #AAB8C2; /* Gray for borders */
            --success: #17BF63; /* Green for feedback */
            --error: #E0245E; /* Red for errors */
            --bubble-bg: #3B4446; /* Assistant bubble */
            --input-bg: #2A2F31; /* Input area */
            --shadow: rgba(0, 0, 0, 0.2);
        }
        """
    else:
        return """
        :root {
            --bg: #F7F7F7; /* visitethiopia.et light background */
            --text: #333333; /* Dark text */
            --accent: #005B99; /* Website blue */
            --secondary: #D0D0D0; /* Light gray for borders */
            --success: #28A745; /* Website green */
            --error: #D32F2F; /* Website red */
            --bubble-bg: #FFFFFF; /* White bubble */
            --input-bg: #FFFFFF; /* Input area */
            --shadow: rgba(0, 0, 0, 0.08);
        }
        """

st.markdown(
    f"""
    <style>
    {get_theme_css()}
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Inter', 'Helvetica', sans-serif;
    }}

    /* Floating Action Button */
    .chat-fab {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50px;
        height: 50px;
        background: var(--accent);
        color: var(--text);
        border: none;
        border-radius: 50%;
        box-shadow: 0 4px 12px var(--shadow);
        cursor: pointer;
        font-size: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .chat-fab:hover {{
        transform: scale(1.1);
        box-shadow: 0 6px 16px var(--shadow);
        background: var(--success);
    }}

    /* Chat Popup */
    .chat-container {{
        position: fixed;
        bottom: 80px;
        right: 20px;
        width: 360px;
        height: 480px;
        background: var(--input-bg);
        border-radius: 12px;
        box-shadow: 0 8px 24px var(--shadow);
        border: 1px solid var(--secondary);
        z-index: 999;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }}

    /* Chat Header */
    .chat-header {{
        background: var(--bg);
        color: var(--text);
        padding: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 16px;
        border-bottom: 1px solid var(--secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .theme-toggle {{
        background: var(--bubble-bg);
        border: 1px solid var(--secondary);
        color: var(--text);
        cursor: pointer;
        font-size: 18px;
        padding: 6px;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease;
    }}
     .theme-toggle:hover {{
        transform: scale(1.1);
        background: var(--accent);
    }}

    /* Chat Body */
    .chat-body {{
        flex: 1;
        overflow-y: auto;
        padding: 12px;
        background: var(--input-bg);
        display: flex;
        flex-direction: column;
        gap: 10px;
        scroll-behavior: smooth;
    }}

    /* Messages */
    .assistant {{
        background: var(--bubble-bg);
        color: var(--text);
        padding: 10px 14px;
        border-radius: 12px 12px 12px 4px;
        max-width: 80%;
        align-self: flex-start;
        line-height: 1.5;
        box-shadow: 0 1px 3px var(--shadow);
    }}
    .user {{
        background: var(--accent);
        color: var(--text);
        padding: 10px 14px;
        border-radius: 12px 12px 4px 12px;
        max-width: 80%;
        align-self: flex-end;
        line-height: 1.5;
        box-shadow: 0 1px 3px var(--shadow);
    }}

    /* Suggested Questions */
    .suggestion-btn {{
        background: var(--bubble-bg);
        border: 1px solid var(--secondary);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        cursor: pointer;
        color: var(--text);
        font-size: 13px;
        transition: all 0.2s ease;
    }}
    .suggestion-btn:hover {{
        background: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px var(--shadow);
    }}

    /* Input Area */
    .input-area {{
        display: flex;
        padding: 10px;
        background: var(--input-bg);
        border-top: 1px solid var(--secondary);
        gap: 8px;
    }}
    .input-area input {{
        flex: 1;
        padding: 8px 12px;
        border: 1px solid var(--secondary);
        border-radius: 8px;
        outline: none;
        font-size: 13px;
        background: var(--bubble-bg);
        color: var(--text);
    }}
    .input-area input:focus {{
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(29, 161, 242, 0.2);
    }}
    .input-area button {{
        background: var(--accent);
        color: var(--text);
        border: none;
        padding: 8px 14px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 13px;
    }}
    .input-area button:hover {{
        background: var(--success);
    }}

    /* Typing Indicator */
    .typing-indicator {{
        display: flex;
        padding: 8px 12px;
        background: var(--bubble-bg);
        border-radius: 10px;
        align-self: flex-start;
        width: fit-content;
    }}
    .typing-dot {{
        width: 6px;
        height: 6px;
        background: var(--accent);
        border-radius: 50%;
        margin: 0 2px;
        animation: pulse 1s infinite ease-in-out;
    }}
    .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.4; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1); }}
    }}

    /* Feedback Buttons */
    .feedback-container {{
        display: flex;
        justify-content: flex-start;
        gap: 8px;
        margin-top: 6px;
    }}
    .feedback-btn {{
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .helpful-btn {{
        background: var(--success);
        color: var(--text);
        border: none;
    }}
    .helpful-btn:hover {{
        background: darken(var(--success), 10%);
        transform: scale(1.05);
    }}
    .not-helpful-btn {{
        background: var(--bubble-bg);
        color: var(--error);
        border: 1px solid var(--error);
    }}
    .not-helpful-btn:hover {{
        background: var(--error);
        color: var(--text);
        transform: scale(1.05);
    }}

    /* Error Alert */
    .error-alert {{
        background: var(--bubble-bg);
        color: var(--error);
        padding: 8px 12px;
        border: 1px solid var(--error);
        border-radius: 8px;
        font-size: 13px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .chat-container {{
            width: 90vw;
            height: 70vh;
            right: 5vw;
        }}
        .chat-fab {{
            width: 45px;
            height: 45px;
            font-size: 20px;
        }}
        .chat-header {{
            font-size: 14px;
        }}
        .suggestion-btn, .input-area input, .input-area button {{
            font-size: 12px;
        }}
        .feedback-btn {{
            padding: 6px 10px;
            font-size: 12px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Feedback Function

def send_feedback(question: str, answer: str, helpful: bool):
    try:
        response = requests.post(
            FEEDBACK_ENDPOINT,
            json={"question": question, "answer": answer, "was_helpful": helpful},
            timeout=5
        )
        print(f"Feedback sent: {response.status_code}, {response.text}")
        return True
    except Exception as e:
        print(f"Feedback error: {e}")
        return False

# Backend Communication

def get_answer(query: str) -> str:
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            print(f"Sending query to backend (attempt {attempt + 1}): {query}")
            response = requests.post(ASK_ENDPOINT, json={"query": query}, timeout=90)
            print(f"Backend response status: {response.status_code}, {response.text}")
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "I couldn't find an answer.")
                answer = re.sub(r'https?://[^\s]+', '', answer).strip()
                answer = re.sub(r'Source:.*', '', answer).strip()
                answer = re.sub(r'\[URL\]', '', answer).strip()
                return answer
            else:
                return f"Error {response.status_code}: {response.text}"
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return "Cannot connect to backend. Please check if FastAPI is running."
        except requests.exceptions.Timeout as e:
            print(f"Timeout error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return "Request timed out after 90 seconds. Try again later."
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}): {e}")
            return f"Connection error: {str(e)}"
        time.sleep(retry_delay)
    return "Failed to get response after multiple attempts."

# Main UI Logic

if not st.session_state.chat_open:
    if st.button("💬 Chat Assitance ", key="open_chat", help="Open Travel Assistant"):
        st.session_state.chat_open = True
        st.rerun()

if st.session_state.chat_open:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">Visit Ethiopia Assistant 🌍', unsafe_allow_html=True)
    if st.button("Toggle Theme", key="theme_toggle", help="Switch between dark and light theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-body">', unsafe_allow_html=True)

    # Greeting
    if not st.session_state.greeting_shown:
        st.markdown(
            '<div class="assistant">Welcome to Visit Ethiopia! Ask about UNESCO sites, hotels, or tours.</div>',
            unsafe_allow_html=True
        )
        st.session_state.greeting_shown = True

    # Error Alert
    if st.session_state.error_message:
        st.markdown(
            f'<div class="error-alert">{st.session_state.error_message}'
            '<button onclick="this.parentElement.style.display=\'none\'" style="border:none;background:none;color:var(--error);cursor:pointer;">✖</button></div>',
            unsafe_allow_html=True
        )

    # Messages
    for idx, msg in enumerate(st.session_state.messages):
        role_class = "user" if msg["role"] == "user" else "assistant"
        content_hash = hashlib.md5(msg["content"].encode()).hexdigest()[:10]
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg["content"] not in st.session_state.feedback_given:
            st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍 Helpful", key=f"helpful_{idx}_{content_hash}"):
                    if send_feedback(st.session_state.last_query, msg["content"], True):
                        st.session_state.feedback_given[msg["content"]] = True
                        st.rerun()
            with col2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{idx}_{content_hash}"):
                    if send_feedback(st.session_state.last_query, msg["content"], False):
                        st.session_state.feedback_given[msg["content"]] = True
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown('<div class="assistant">Thanks for your feedback!</div>', unsafe_allow_html=True)

    # Typing Indicator
    if st.session_state.waiting:
        st.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>'
            '</div>',
            unsafe_allow_html=True
        )

    # Suggested Questions
    if not st.session_state.messages:
        st.markdown('<div style="font-weight:600;font-size:13px;color:var(--text);margin:8px 0;">Suggested Questions:</div>', unsafe_allow_html=True)
        suggestions = [
            "What are UNESCO World Heritage Sites in Ethiopia?",
            "Tell me about hotels in Addis Ababa",
            "Details of the Bale Mountains National Park Tour"
        ]
        for q in suggestions:
            if st.button(q, key=f"sug_{hash(q)}", help="Ask this question"):
                st.session_state.last_query = q
                st.session_state.waiting = True
                st.session_state.input_key += 1
                st.session_state.error_message = None
                st.rerun()

    # Input Area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask about Ethiopia...",
            key=f"user_input_{st.session_state.input_key}",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("Send", key="send_button"):
            if user_input.strip():
                st.session_state.last_query = user_input
                st.session_state.waiting = True
                st.session_state.input_key += 1
                st.session_state.error_message = None
                st.rerun()
            else:
                st.session_state.error_message = "Please enter a question."
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Close Button
    if st.button("Close", key="close_chat"):
        st.session_state.chat_open = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Backend Call

if st.session_state.waiting:
    query = st.session_state.last_query
    if query:
        with st.spinner("Fetching answer..."):
            st.session_state.messages.append({"role": "user", "content": query})
            answer = get_answer(query)
            if answer.startswith("❌"):
                st.session_state.error_message = answer
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.waiting = False
            st.rerun()

# Auto-scroll
st.markdown(
    """
    <script>
    parent = window.parent.document || window.document;
    chat_body = parent.querySelector('.chat-body');
    if (chat_body) {
        chat_body.scrollTop = chat_body.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
)