"""
Streamlit Materials Science Chat Interface
ChatGPT-like interface for materials science conversations
"""

import streamlit as st
import time
from datetime import datetime
from chat_agent import ChatAgent, ChatConfig
import os


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = ChatAgent()
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False


def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        # Clean the content to remove any unwanted tags
        cleaned_content = clean_message_content(content)
        st.markdown(cleaned_content)


def clean_message_content(content: str) -> str:
    """Clean message content to remove unwanted tags and formatting"""
    if not content:
        return content
    
    # Remove any remaining User: or Assistant: tags at the beginning
    content = content.strip()
    
    # Remove leading "Assistant:" if present
    if content.startswith('Assistant:'):
        content = content[10:].strip()
    
    # Remove leading "User:" if present (shouldn't happen but just in case)
    if content.startswith('User:'):
        content = content[5:].strip()
    
    # Remove [Response] tags
    content = content.replace('[Response]', '').strip()
    
    # Split into lines and clean
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that start with User: or Assistant: or are empty
        if line.startswith(('User:', 'Assistant:')) or not line:
            continue
        # Skip repetitive patterns
        if line == '[Response]':
            continue
        cleaned_lines.append(line)
    
    # Join lines and clean up
    cleaned = '\n'.join(cleaned_lines).strip()
    
    # Remove any trailing User: or Assistant: references
    if cleaned.endswith(('User:', 'Assistant:')):
        cleaned = cleaned.rsplit('User:', 1)[0].rsplit('Assistant:', 1)[0].strip()
    
    return cleaned


def display_welcome_message():
    """Display welcome message and example questions"""
    st.markdown("""
    ## 🧪 LLaMat-Chat - Materials Science Chat Assistant
    
    Welcome! I'm your AI assistant specialized in materials science. I can help you with:
    
    - **Materials Chemistry**: Chemical compositions, synthesis methods, and reactions
    - **Characterization**: XRD, SEM, TEM, spectroscopy, and other analytical techniques
    - **Properties**: Electronic, optical, mechanical, and thermal properties
    - **Applications**: Industrial uses, device applications, and performance metrics
    - **Recent Advances**: Latest research trends and breakthrough discoveries
    
    ### 💡 Example Questions:
    - "What are the key properties of perovskite materials?"
    - "How does sol-gel synthesis work for TiO2 nanoparticles?"
    - "Explain the difference between crystalline and amorphous materials"
    - "What are the applications of graphene in electronics?"
    - "Describe the crystal structure of diamond and its properties"
    
    **Start by typing your question below!** 👇
    """)


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Materials Science Chat",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for ChatGPT-like styling
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff;
        border-left: 4px solid #ff7f0e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #28a745;
    }
    
    .status-offline {
        background-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🧪 Materials Science Chat</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Powered by LLaMat-2-Chat Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Conversation Info")
        
        # Model status
        try:
            # Simple ping to check if model is online
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                st.markdown("""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span class="status-indicator status-online"></span>
                        <strong>Model Status</strong>
                    </div>
                    <p style="margin: 0; color: #28a745;">🟢 Online</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span class="status-indicator status-offline"></span>
                        <strong>Model Status</strong>
                    </div>
                    <p style="margin: 0; color: #dc3545;">🔴 Offline</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-indicator status-offline"></span>
                    <strong>Model Status</strong>
                </div>
                <p style="margin: 0; color: #dc3545;">🔴 Offline</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Conversation metrics
        summary = st.session_state.chat_agent.get_conversation_summary()
        st.markdown(f"""
        <div class="metric-card">
            <strong>Messages</strong><br>
            <span style="font-size: 1.5rem; color: #1f77b4;">{summary['total_messages']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Tokens Used</strong><br>
            <span style="font-size: 1.5rem; color: #ff7f0e;">{summary['current_tokens']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Cache info
        cache_files = len([f for f in os.listdir("cache") if f.startswith("chat_cache_")]) if os.path.exists("cache") else 0
        st.markdown(f"""
        <div class="metric-card">
            <strong>Cached Conversations</strong><br>
            <span style="font-size: 1.5rem; color: #2ca02c;">{cache_files}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Controls
        st.markdown("### ⚙️ Controls")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_agent.clear_history()
            st.session_state.messages = []
            st.session_state.conversation_started = False
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            conversation = st.session_state.chat_agent.export_conversation()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create downloadable text file
            chat_text = f"Materials Science Chat Export - {timestamp}\n"
            chat_text += "=" * 50 + "\n\n"
            
            for msg in conversation:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                label="Download Chat History",
                data=chat_text,
                file_name=f"materials_science_chat_{timestamp}.txt",
                mime="text/plain"
            )
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask specific questions about materials
        - Request detailed explanations
        - Ask for examples and applications
        - Follow up with related questions
        - Use scientific terminology when appropriate
        """)
    
    # Main chat interface
    chat_container = st.container()
    
    # Display welcome message if no conversation started
    if not st.session_state.conversation_started and len(st.session_state.messages) == 0:
        display_welcome_message()
    else:
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
    
    # Chat input at the bottom
    with st.container():
        user_input = st.chat_input(
            "Ask me anything about materials science...",
            key="chat_input"
        )
        
        # Handle chat input
        if user_input:
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            display_chat_message("user", user_input)
            
            # Show typing indicator
            with st.spinner("Thinking..."):
                try:
                    # Get response from chat agent
                    response = st.session_state.chat_agent.chat(user_input)
                    
                    # Clean the response to ensure no unwanted tags
                    cleaned_response = clean_message_content(response)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
                    display_chat_message("assistant", cleaned_response)
                    
                    st.session_state.conversation_started = True
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    display_chat_message("assistant", error_msg)
                    st.error(f"Error: {e}")
            
            # Clear the input and rerun
            st.rerun()


if __name__ == "__main__":
    main()
