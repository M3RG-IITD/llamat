#!/usr/bin/env python3
"""
Interactive NER Agent - Simple command-line interface for Named Entity Recognition
"""

import json
from ner_agent import NERAgent, NERConfig


def print_header():
    """Print welcome header"""
    print("="*60)
    print("           NAMED ENTITY RECOGNITION AGENT")
    print("="*60)
    print()


def print_entity_types():
    """Print available entity types"""
    print("Common entity types:")
    entity_examples = {
        "PERSON": "Names of people (e.g., John Smith, Mary Johnson)",
        "ORGANIZATION": "Companies, institutions (e.g., Apple Inc., Harvard University)",
        "LOCATION": "Places, cities, countries (e.g., New York, California, USA)",
        "DATE": "Dates and times (e.g., January 2023, 1995, yesterday)",
        "MONEY": "Monetary amounts (e.g., $100, 500 dollars, €50)",
        "PRODUCT": "Products, goods (e.g., iPhone, Tesla Model S)",
        "EVENT": "Named events (e.g., World War II, Olympic Games)",
        "TECHNOLOGY": "Tech terms (e.g., AI, machine learning, blockchain)"
    }
    
    for entity, description in entity_examples.items():
        print(f"  • {entity}: {description}")
    print()


def get_entity_types():
    """Get entity types from user input"""
    print("Enter the entity types you want to extract (comma-separated):")
    print("Example: PERSON, ORGANIZATION, LOCATION, DATE, MONEY")
    
    while True:
        user_input = input("Entity types: ").strip()
        if not user_input:
            print("Please enter at least one entity type.")
            continue
            
        entities = [entity.strip().upper() for entity in user_input.split(",")]
        if not entities:
            print("Please enter valid entity types.")
            continue
            
        # Validate entities
        valid_entities = []
        for entity in entities:
            if entity and entity.replace("_", "").isalnum():
                valid_entities.append(entity)
            else:
                print(f"Warning: '{entity}' is not a valid entity name. Skipping.")
        
        if valid_entities:
            return valid_entities
        else:
            print("No valid entities provided. Please try again.")


def get_text_input():
    """Get text input from user"""
    print("\nEnter the text paragraph to analyze:")
    print("(Press Ctrl+D (Unix/Mac) or Ctrl+Z (Windows) when finished)")
    print("-" * 40)
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines).strip()


def display_results(result):
    """Display extraction results in a nice format"""
    print("\n" + "="*60)
    print("                    EXTRACTION RESULTS")
    print("="*60)
    
    if not any(result.values()):
        print("No entities found in the text.")
        return
    
    for entity_type, entities in result.items():
        if entities:
            print(f"\n{entity_type}:")
            for i, entity in enumerate(entities, 1):
                print(f"  {i}. {entity}")
        else:
            print(f"\n{entity_type}: (No entities found)")
    
    print("\n" + "-"*60)
    print("JSON Format:")
    print(json.dumps(result, indent=2))


def main():
    """Main interactive loop"""
    print_header()
    print_entity_types()
    
    # Get entity types
    entities = get_entity_types()
    print(f"\nSelected entities: {', '.join(entities)}")
    
    # Initialize agent
    try:
        print("\nInitializing NER Agent...")
        ner_agent = NERAgent(entities)
        print("✓ Agent initialized successfully!")
    except Exception as e:
        print(f"✗ Error initializing agent: {e}")
        print("Make sure your local model is running at http://localhost:8000")
        return
    
    while True:
        print("\n" + "="*60)
        
        # Get text input
        text = get_text_input()
        
        if not text:
            print("No text provided. Exiting...")
            break
        
        print(f"\nAnalyzing text ({len(text)} characters)...")
        
        try:
            # Extract entities
            result = ner_agent.extract_entities(text)
            
            # Display results
            display_results(result)
            
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
        
        # Ask if user wants to continue
        print("\n" + "-"*60)
        continue_choice = input("Analyze another text? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("\nThank you for using the NER Agent! 👋")


if __name__ == "__main__":
    main()
