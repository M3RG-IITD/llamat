#!/usr/bin/env python3
"""
Simple example demonstrating the NER Agent usage
"""

from ner_agent import NERAgent, NERConfig


def main():
    """Simple example of using the NER Agent"""
    
    # Define the entities you want to extract
    entities = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"]
    
    # Initialize the NER agent
    ner_agent = NERAgent(entities)
    
    # Example text for entity extraction
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California on April 1, 1976. 
    The company is headquartered in Cupertino and has offices worldwide. In 2023, Apple reported revenue of $394.3 billion.
    Tim Cook is the current CEO of Apple, succeeding Steve Jobs in August 2011.
    The iPhone was first introduced on January 9, 2007, revolutionizing the smartphone industry.
    """
    
    print("Input Text:")
    print(text)
    print("\n" + "="*50)
    
    # Extract entities
    print("Extracting entities...")
    result = ner_agent.extract_entities(text)
    
    # Display results
    print("\nExtracted Entities:")
    for entity_type, entities_list in result.items():
        if entities_list:
            print(f"{entity_type}: {entities_list}")
        else:
            print(f"{entity_type}: (No entities found)")
    
    print("\n" + "="*50)
    print("JSON Output:")
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
