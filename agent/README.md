# Named Entity Recognition Agent

A simple information extraction agent that uses your local LLM model to perform Named Entity Recognition (NER) tasks.

## Features

- ✅ Configurable entity types for extraction
- ✅ Uses your local LLM model at `http://localhost:8000`
- ✅ Returns structured JSON output
- ✅ Simple and easy-to-use interface
- ✅ Batch processing support
- ✅ Interactive command-line interface

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your local LLM model is running at `http://localhost:8000`

## Usage

### 1. Simple Example

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

### 2. Run the Simple Example

```bash
python simple_example.py
```

### 3. Interactive Mode

```bash
python interactive_ner.py
```

This will start an interactive session where you can:
- Specify which entities to extract
- Input text paragraphs
- View results in a user-friendly format

### 4. Materials Science Example

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

### 5. Interactive Chat Interface

Launch the ChatGPT-like materials science chat:
```bash
python launch_chat.py
```

Or directly:
```bash
streamlit run streamlit_chat.py
```

**Features:**
- Interactive materials science conversations
- Follow-up question support
- Automatic chat history caching
- Beautiful web interface
- Export conversation history

### 6. Command Line Chat

For terminal-based chat:
```bash
python chat_agent.py
```

### 7. Custom Configuration

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

## Requirements

- Python 3.7+
- `requests` library
- Local LLM model running on `http://localhost:8000`

## Notes

- The agent uses the `/v1/completions` endpoint of your local model
- Results are returned as JSON with entity types as keys and lists of extracted entities as values
- If no entities of a particular type are found, an empty list is returned for that type
- The agent handles parsing errors gracefully and returns empty results on failure
- Empty strings are automatically filtered out from the results
