import streamlit as st
import requests
import json
import logging
from datetime import datetime
import re
import time
import hashlib

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced CSS with dark mode support and better accessibility
st.markdown("""
    <style>
    :root {
        --primary: #4CAF50;
        --secondary: #2E7D32;
        --error: #D32F2F;
        --text: #333333;
        --bg: #FFFFFF;
        --card-bg: #F5F5F5;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text: #E0E0E0;
            --bg: #121212;
            --card-bg: #1E1E1E;
        }
    }
    
    .chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: var(--primary);
        color: white;
        padding: 15px;
        border-radius: 50%;
        cursor: pointer;
        z-index: 1000;
        font-size: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .chat-button:hover {
        transform: scale(1.1);
        background-color: var(--secondary);
    }
    
    .chat-window {
        position: fixed;
        bottom: 80px;
        right: 20px;
        width: 90%;
        max-width: 400px;
        max-height: 60vh;
        background-color: var(--card-bg);
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        z-index: 1000;
        overflow-y: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: var(--text);
    }
    
    .chat-input {
        position: fixed;
        bottom: 10px;
        right: 20px;
        width: 90%;
        max-width: 400px;
        z-index: 1000;
    }
    
    .close-button {
        background-color: var(--error);
        color: white;
        border-radius: 5px;
        padding: 8px;
        text-align: center;
        cursor: pointer;
        margin-top: 10px;
    }
    
    .error-message {
        color: var(--error);
        font-size: 14px;
        margin-top: 10px;
    }
    
    .typing-indicator {
        display: flex;
        padding: 10px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: var(--text);
        border-radius: 50%;
        margin: 0 3px;
        animation: typingAnimation 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typingAnimation {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    /* Accessibility improvements */
    [role="dialog"] {
        outline: none;
    }
    
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border-width: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with enhancements
if "chat_visible" not in st.session_state:
    st.session_state.chat_visible = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None
if "user_id" not in st.session_state:
    # Generate anonymous user ID for analytics
    st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "conversation_start_time" not in st.session_state:
    st.session_state.conversation_start_time = None

# Enhanced input sanitization with more comprehensive checks
def sanitize_input(query: str) -> str:
    query = query.strip()[:500]  # Limit length
    
    # Remove potentially harmful characters while allowing more language support
    query = re.sub(r'[<>{}[\]\\;]', '', query)
    
    # Check for suspicious patterns
    if re.search(r'(http|ftp|https):\/\/', query, re.IGNORECASE):
        raise ValueError("Links are not allowed in queries")
        
    if len(query) < 2:
        raise ValueError("Query too short")
        
    return query

# Track conversation metrics
def log_conversation_metrics():
    if st.session_state.messages:
        duration = (datetime.now() - st.session_state.conversation_start_time).total_seconds() if st.session_state.conversation_start_time else 0
        logger.info(
            f"Conversation metrics - User: {st.session_state.user_id}, "
            f"Messages: {len(st.session_state.messages)}, "
            f"Duration: {duration:.2f}s"
        )

# Welcome message with clear purpose
WELCOME_MESSAGE = """
**Welcome to Ethiopia Travel Assistant!** ✈️🌍

I can help you with:
- Destination spaces and tourist sites
- Tour packages and itineraries
- Hotel recommendations
- Travel planning advice
- Visa and stopover information

Try asking:  
_"What are the top destination spaces in Ethiopia?"_  
or  
_"Tell me about 3-day tour packages"_
"""

# Floating chat button
if not st.session_state.chat_visible:
    if st.button("🗨️", key="chat_button", help="Chat with Ethiopia Travel Assistant"):
        st.session_state.chat_visible = True
        st.session_state.conversation_start_time = datetime.now()
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "assistant", "content": WELCOME_MESSAGE})
        st.rerun()
else:
    # Chat window with enhanced features
    with st.container():
        st.markdown('<div class="chat-window" role="dialog" aria-labelledby="chat-title">', unsafe_allow_html=True)
        st.markdown('<span id="chat-title" class="sr-only">Chat with Ethiopia Travel Assistant</span>', unsafe_allow_html=True)
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Show typing indicator when waiting for response
        if st.session_state.waiting_for_response:
            with st.chat_message("assistant"):
                st.markdown(
                    '<div class="typing-indicator">'
                    '<div class="typing-dot"></div>'
                    '<div class="typing-dot"></div>'
                    '<div class="typing-dot"></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
        
        # Display error message if any
        if st.session_state.error:
            st.markdown(f'<div class="error-message" role="alert">{st.session_state.error}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced chat input with analytics
    with st.container():
        prompt = st.chat_input(
            "Ask about tours, destinations, or travel info...",
            key="chat_input",
            on_submit=lambda: log_conversation_metrics()
        )
        
        if prompt:
            try:
                prompt = sanitize_input(prompt)
                logger.info(f"User query from {st.session_state.user_id}: {prompt}")
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.error = None
                st.session_state.waiting_for_response = True
                st.rerun()

                with st.spinner(""):
                    for attempt in range(3):
                        try:
                            # Use environment variable for backend URL
                            backend_url = st.secrets.get("BACKEND_URL", "http://localhost:8000")
                            
                            # Add analytics headers
                            headers = {
                                "X-User-ID": st.session_state.user_id,
                                "X-Session-Start": st.session_state.conversation_start_time.isoformat() if st.session_state.conversation_start_time else ""
                            }
                            
                            response = requests.post(
                                f"{backend_url}/ask",
                                json={"question": prompt},
                                headers=headers,
                                timeout=30  # More reasonable timeout
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                answer = data["answer"]
                                
                                # Log response metrics
                                logger.info(
                                    f"Response received - User: {st.session_state.user_id}, "
                                    f"Processing time: {data.get('processing_time', 0):.2f}s, "
                                    f"Chars: {len(answer)}"
                                )
                                
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                break
                                
                            elif response.status_code == 400:
                                st.session_state.error = "Please ask a more specific question."
                                break
                                
                            elif response.status_code == 429:
                                st.session_state.error = "I'm getting too many requests. Please try again in a minute."
                                break
                                
                            else:
                                st.session_state.error = "Sorry, I'm having trouble answering right now. Please try again later."
                                break
                                
                        except requests.Timeout:
                            if attempt == 2:
                                st.session_state.error = "The request timed out. Please try again with a simpler question."
                                logger.error(f"Timeout for user {st.session_state.user_id}")
                            time.sleep(1)
                            
                        except Exception as e:
                            st.session_state.error = "An unexpected error occurred."
                            logger.error(f"Unexpected error for user {st.session_state.user_id}: {str(e)}")
                            break
                            
            except ValueError as ve:
                st.session_state.error = str(ve)
                logger.warning(f"Invalid input from user {st.session_state.user_id}: {ve}")
                
            finally:
                st.session_state.waiting_for_response = False
                st.rerun()

    # Close button with confirmation
    if st.button("Close Chat", key="close_button"):
        log_conversation_metrics()
        st.session_state.chat_visible = False
        st.rerun()