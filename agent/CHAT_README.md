# 🗣️ Chat Agent - Quick Reference

> **📚 For complete documentation, see [README.md](README.md) which covers both Chat and NER agents.**

This file provides a quick reference for the Chat Agent functionality. For the full documentation including NER agent capabilities, installation instructions, and comprehensive troubleshooting, please refer to the main [README.md](README.md).

## Quick Start

### 1. Launch Web Interface (Recommended)
```bash
python launch_chat.py
```

### 2. Command Line Chat
```bash
python chat_agent.py
```

### 3. Programmatic Usage
```python
from chat_agent import ChatAgent

agent = ChatAgent()
response = agent.chat("What are perovskite materials?")
print(response)
```

## Key Features

- ✅ **Interactive Chat**: Materials science Q&A with context
- ✅ **Streamlit Interface**: Beautiful ChatGPT-like web UI
- ✅ **Auto-caching**: Conversation history management
- ✅ **Export**: Download conversation history
- ✅ **Model Monitoring**: Real-time availability checking

## Quick Commands

**In terminal chat:**
- `clear` - Start new conversation
- `summary` - Show conversation stats
- `quit` / `exit` / `bye` - End session

## Configuration Example

```python
from chat_agent import ChatAgent, ChatConfig

config = ChatConfig(
    temperature=0.7,        # Creativity (0.0-1.0)
    max_tokens=512,         # Response length
    max_context_tokens=3000 # Context window
)

agent = ChatAgent(config)
```

## Troubleshooting Quick Tips

1. **Model not available**: Ensure llamat-2 is running at `http://localhost:8000`
2. **Streamlit issues**: Try `streamlit run streamlit_chat.py --server.port 8502`
3. **Memory issues**: Reduce `max_context_tokens` or clear cache
4. **Cache problems**: Delete `cache/` directory

## 📖 Full Documentation

For complete information including:
- NER Agent functionality
- Detailed configuration options
- Advanced features
- Comprehensive troubleshooting
- API reference
- Performance tips

**👉 See [README.md](README.md)**
