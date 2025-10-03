# 🧪 LLaMAT Agent Suite

A comprehensive agent suite for materials science applications, featuring both interactive chat capabilities and Named Entity Recognition (NER) functionality using your local llamat-2-chat model.

## Features

### Chat Agent
- ✅ **Interactive Chat**: Ask materials science questions and get detailed answers
- ✅ **Follow-up Questions**: Maintain conversation context for related questions
- ✅ **Chat History Management**: Automatic caching when token limits are reached
- ✅ **Streamlit Interface**: Beautiful ChatGPT-like web interface
- ✅ **Command Line Interface**: Simple terminal-based chat for quick interactions
- ✅ **Export Functionality**: Download conversation history
- ✅ **Model Status Monitoring**: Real-time model availability checking

### NER Agent
- ✅ **Configurable Entity Types**: Extract custom entity types from text
- ✅ **Materials Science Entities**: Pre-configured materials science entity types
- ✅ **Structured JSON Output**: Clean, structured extraction results
- ✅ **Batch Processing**: Process multiple texts at once
- ✅ **Interactive Interface**: Command-line interface for easy testing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your local llamat-2 model is running at `http://localhost:8000`

## Usage

## 🗣️ Chat Agent

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

### 2. Command Line Chat Interface

For quick terminal-based interactions:
```bash
python chat_agent.py
```

**Commands:**
- Type your question and press Enter
- Type `clear` to start a new conversation
- Type `summary` to see conversation info
- Type `quit`, `exit`, or `bye` to end

### 3. Programmatic Chat Usage

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

## 🏷️ NER Agent

### 1. Simple NER Example

```python
from ner_agent import NERAgent

# Define entities to extract
entities = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"]

# Initialize the agent
ner_agent = NERAgent(entities)

# Extract entities from text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976."
result = ner_agent.extract_entities(text)

print(result)
# Output: {
#     "PERSON": ["Steve Jobs"],
#     "ORGANIZATION": ["Apple Inc."],
#     "LOCATION": ["Cupertino", "California"],
#     "DATE": ["April 1, 1976"],
#     "MONEY": []
# }
```

### 2. Run the Simple NER Example

```bash
python simple_example.py
```

### 3. Interactive NER Mode

```bash
python interactive_ner.py
```

This will start an interactive session where you can:
- Specify which entities to extract
- Input text paragraphs
- View results in a user-friendly format

### 4. Materials Science NER

```bash
python materials_science_ner.py
```

For materials science specific entity extraction with pre-configured tags:
- `APL` - Application
- `MAT` - Material  
- `CMT` - Characterization
- `DSC` - Sample description
- `PRO` - Property
- `SMT` - Synthesis method
- `SPL` - Symmetry/phase label

Interactive materials science mode:
```bash
python materials_science_ner.py --interactive
```

### 5. Custom NER Configuration

```python
from ner_agent import NERAgent, NERConfig

# Custom configuration
config = NERConfig(
    model_endpoint="http://localhost:8000/v1/completions",
    temperature=0.1,
    max_tokens=512
)

# Initialize with custom config
ner_agent = NERAgent(["PERSON", "ORG"], config)
```

## ⚙️ Configuration

### Chat Agent Configuration

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

### NER Agent Configuration

```python
from ner_agent import NERAgent, NERConfig

config = NERConfig(
    model_endpoint="http://localhost:8000/v1/completions",
    model_name="llamat-2-chat",
    temperature=0.1,  # Lower temperature for more consistent extraction
    max_tokens=512
)

ner_agent = NERAgent(["PERSON", "ORG", "LOCATION"], config)
```

## 💬 Chat History Management

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

## Available Entity Types

You can define any entity types you want. Common examples include:

### General Entities
- `PERSON` - Names of people
- `ORGANIZATION` - Companies, institutions
- `LOCATION` - Places, cities, countries
- `DATE` - Dates and times
- `MONEY` - Monetary amounts
- `PRODUCT` - Products, goods
- `EVENT` - Named events
- `TECHNOLOGY` - Tech terms

### Materials Science Entities (Pre-configured)
- `APL` - Application (use cases, applications, purposes)
- `MAT` - Material (chemical compounds, elements, materials, molecules)
- `CMT` - Characterization (analysis techniques, measurement methods)
- `DSC` - Sample description (sample composition, structure, morphology)
- `PRO` - Property (physical properties, chemical properties, performance metrics)
- `SMT` - Synthesis method (preparation methods, synthesis techniques)
- `SPL` - Symmetry/phase label (crystal structures, phases, symmetry groups)

## 💡 Example Conversations

### Materials Science Chat Examples

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

### NER Extraction Examples

**Text:** "Silicon carbide (SiC) is synthesized using chemical vapor deposition (CVD) for high-temperature applications in power electronics."

**Entities Extracted:**
```json
{
    "MAT": ["Silicon carbide", "SiC"],
    "SMT": ["chemical vapor deposition", "CVD"],
    "APL": ["high-temperature applications", "power electronics"],
    "PRO": [],
    "CMT": [],
    "DSC": [],
    "SPL": []
}
```

## API Reference

### NERAgent Class

#### `__init__(entities, config=None)`
- `entities`: List of entity types to extract
- `config`: Optional NERConfig object for custom settings

#### `extract_entities(text)`
- `text`: Input text paragraph
- Returns: Dictionary with entity types as keys and lists of extracted entities as values

#### `extract_entities_batch(texts)`
- `texts`: List of text paragraphs
- Returns: List of dictionaries with extracted entities for each text

### NERConfig Class

- `model_endpoint`: URL of your local model API (default: `http://localhost:8000/v1/completions`)
- `model_name`: Name of the model to use
- `temperature`: Sampling temperature (0.0 to 1.0)
- `max_tokens`: Maximum tokens in response

### ChatAgent Class

#### `__init__(config=None)`
- `config`: Optional ChatConfig object for custom settings

#### `chat(message)`
- `message`: User message/question
- Returns: Model response as string

#### `clear_conversation()`
- Clears current conversation history

#### `get_conversation_summary()`
- Returns: Dictionary with conversation statistics

#### `export_conversation()`
- Returns: Complete conversation history

## 📁 File Structure

```
agent/
├── chat_agent.py          # Core chat agent class
├── streamlit_chat.py      # Streamlit web interface
├── launch_chat.py         # Launch script with checks
├── ner_agent.py           # Core NER agent class
├── interactive_ner.py     # Interactive NER interface
├── materials_science_ner.py  # Materials science NER examples
├── simple_example.py      # Simple NER example
├── requirements.txt       # Dependencies
├── README.md             # This unified documentation
├── CHAT_README.md        # Legacy chat documentation
└── cache/                # Conversation cache directory
    ├── chat_cache_abc123.pkl
    └── chat_cache_def456.pkl
```

## 🔧 Troubleshooting

### Model Not Available
If you see "Model server is not available":
1. Make sure your llamat-2 model is running
2. Check that it's accessible at `http://localhost:8000`
3. Verify the model endpoint in configuration
4. Test the endpoint: `curl http://localhost:8000/v1/completions`

### Streamlit Issues
If Streamlit doesn't launch:
1. Install Streamlit: `pip install streamlit`
2. Check port availability (default: 8501)
3. Try a different port: `streamlit run streamlit_chat.py --server.port 8502`

### NER Extraction Issues
If NER returns empty results:
1. Check that your model is running
2. Verify the entity types are appropriate for your text
3. Try lowering the temperature for more consistent results
4. Check the model endpoint URL

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

## 🚀 Advanced Features

### Custom System Prompts
Modify the system prompt in `ChatConfig`:

```python
config = ChatConfig(
    system_prompt="You are a materials science expert specializing in nanomaterials..."
)
```

### Batch NER Processing
Process multiple texts:

```python
texts = [
    "Silicon carbide for power electronics...",
    "Perovskite solar cells with high efficiency...",
    "Graphene-based composites for aerospace..."
]

results = ner_agent.extract_entities_batch(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {result}")
```

### Export Conversations
Export chat history from the Streamlit interface or programmatically:

```python
conversation = agent.export_conversation()
# Save to file, database, etc.
```

## 📈 Performance Tips

1. **Token Management**: Keep `max_context_tokens` reasonable (3000-5000)
2. **Cache Cleanup**: Periodically clean old cache files
3. **Model Temperature**: 
   - Chat: Higher values (0.7-0.9) for creative responses
   - NER: Lower values (0.1-0.3) for consistent extraction
4. **Response Length**: Adjust `max_tokens` based on your needs

## 📋 Requirements

- Python 3.7+
- `requests` library for API calls
- `streamlit` for web interface
- `pickle` for conversation caching
- Local llamat-2 model running on `http://localhost:8000`

## 🤝 Contributing

Feel free to enhance the agent suite:
- Add new conversation features
- Improve the Streamlit UI
- Add more materials science examples
- Enhance error handling
- Add new entity types for NER

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify model availability
3. Check logs for error messages
4. Ensure all dependencies are installed

## 📝 Notes

- Both agents use the `/v1/completions` endpoint of your local model
- NER results are returned as JSON with entity types as keys and lists of extracted entities as values
- If no entities of a particular type are found, an empty list is returned for that type
- The agents handle parsing errors gracefully and return empty results on failure
- Empty strings are automatically filtered out from NER results
- Chat conversations are automatically cached when token limits are reached
