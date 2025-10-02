# 🧪 Materials Science Chat Interface

An interactive ChatGPT-like interface for materials science conversations using your local llamat-2 model.

## Features

- ✅ **Interactive Chat**: Ask materials science questions and get detailed answers
- ✅ **Follow-up Questions**: Maintain conversation context for related questions
- ✅ **Chat History Management**: Automatic caching when token limits are reached
- ✅ **Streamlit Interface**: Beautiful ChatGPT-like web interface
- ✅ **Command Line Interface**: Simple terminal-based chat for quick interactions
- ✅ **Export Functionality**: Download conversation history
- ✅ **Model Status Monitoring**: Real-time model availability checking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your local model is running at `http://localhost:8000`

## Usage

### 1. Streamlit Web Interface (Recommended)

Launch the ChatGPT-like web interface:
```bash
python launch_chat.py
```

Or directly:
```bash
streamlit run streamlit_chat.py
```

The interface will open in your browser at `http://localhost:8501`

**Features:**
- Beautiful ChatGPT-like design
- Real-time conversation
- Model status monitoring
- Export chat history
- Conversation metrics

### 2. Command Line Interface

For quick terminal-based interactions:
```bash
python chat_agent.py
```

**Commands:**
- Type your question and press Enter
- Type `clear` to start a new conversation
- Type `summary` to see conversation info
- Type `quit`, `exit`, or `bye` to end

### 3. Programmatic Usage

```python
from chat_agent import ChatAgent, ChatConfig

# Initialize chat agent
agent = ChatAgent()

# Ask a question
response = agent.chat("What are perovskite materials?")
print(response)

# Ask follow-up questions
response = agent.chat("What are their applications?")
print(response)

# Get conversation summary
summary = agent.get_conversation_summary()
print(f"Messages: {summary['total_messages']}, Tokens: {summary['current_tokens']}")
```

## Configuration

Customize the chat agent behavior:

```python
from chat_agent import ChatAgent, ChatConfig

config = ChatConfig(
    model_endpoint="http://localhost:8000/v1/completions",
    temperature=0.7,  # Creativity level (0.0 to 1.0)
    max_tokens=512,   # Max response length
    max_context_tokens=3000,  # Context window size
    cache_dir="cache"  # Directory for conversation cache
)

agent = ChatAgent(config)
```

## Chat History Management

### Automatic Caching
- Conversations are automatically cached when token limits are reached
- Cache files are stored in the `cache/` directory
- Each conversation gets a unique hash-based filename

### Token Management
- Default context limit: 3000 tokens
- System automatically manages context to stay within limits
- Older messages are removed when limit is exceeded
- System message is always preserved

### Cache Files
Cache files are named as: `chat_cache_{hash}.pkl`
- Contains conversation history and token count
- Includes timestamp for reference
- Automatically loaded when needed

## Example Conversations

### Materials Science Questions

**Q:** "What are perovskite materials?"
**A:** "Perovskite materials are characterized by a crystal structure with the formula ABX3, where A and B are cations and X is an anion. They exhibit a cubic cell with space group Pm3m..."

**Q:** "What are their applications?"
**A:** "Perovskite materials are used in applications such as optoelectronics, piezoelectronics, photovoltaics, and thermoelectrics..."

**Q:** "Can you give examples?"
**A:** "Examples include titanium dioxide (TiO2) in photocatalytic applications, lead zirconate titanate (PbZrO3) in piezoelectric devices..."

### Follow-up Questions
The chat maintains context, so you can ask follow-up questions like:
- "What about their toxicity concerns?"
- "Are there alternatives to lead-based materials?"
- "What are the challenges in developing alternatives?"

## File Structure

```
agent/
├── chat_agent.py          # Core chat agent class
├── streamlit_chat.py      # Streamlit web interface
├── launch_chat.py         # Launch script with checks
├── materials_science_ner.py  # NER examples
├── ner_agent.py           # NER agent class
├── requirements.txt       # Dependencies
├── CHAT_README.md        # This file
└── cache/                # Conversation cache directory
    ├── chat_cache_abc123.pkl
    └── chat_cache_def456.pkl
```

## Troubleshooting

### Model Not Available
If you see "Model server is not available":
1. Make sure your llamat-2 model is running
2. Check that it's accessible at `http://localhost:8000`
3. Verify the model endpoint in ChatConfig

### Streamlit Issues
If Streamlit doesn't launch:
1. Install Streamlit: `pip install streamlit`
2. Check port availability (default: 8501)
3. Try a different port: `streamlit run streamlit_chat.py --server.port 8502`

### Cache Issues
If cache files cause problems:
1. Delete the `cache/` directory
2. Restart the chat agent
3. Cache will be recreated automatically

### Memory Issues
If you encounter memory problems:
1. Reduce `max_context_tokens` in ChatConfig
2. Clear conversation history more frequently
3. Delete old cache files

## Advanced Features

### Custom System Prompts
Modify the system prompt in `ChatConfig`:

```python
config = ChatConfig(
    system_prompt="You are a materials science expert specializing in nanomaterials..."
)
```

### Export Conversations
Export chat history from the Streamlit interface or programmatically:

```python
conversation = agent.export_conversation()
# Save to file, database, etc.
```

### Batch Processing
Process multiple questions:

```python
questions = [
    "What are perovskite materials?",
    "What are their properties?",
    "What are their applications?"
]

for question in questions:
    response = agent.chat(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

## Performance Tips

1. **Token Management**: Keep `max_context_tokens` reasonable (3000-5000)
2. **Cache Cleanup**: Periodically clean old cache files
3. **Model Temperature**: Lower values (0.3-0.7) for more focused responses
4. **Response Length**: Adjust `max_tokens` based on your needs

## Contributing

Feel free to enhance the chat interface:
- Add new conversation features
- Improve the Streamlit UI
- Add more materials science examples
- Enhance error handling

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify model availability
3. Check logs for error messages
4. Ensure all dependencies are installed
