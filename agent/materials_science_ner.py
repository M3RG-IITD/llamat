#!/usr/bin/env python3
"""
Materials Science Named Entity Recognition Example
Using llamat-2 model for extracting materials science entities
"""

from ner_agent import NERAgent, NERConfig
import json


def main():
    """Materials Science NER Example"""
    
    # Materials Science Entity Tags with abbreviations
    materials_entities = [
        "APL",  # Application
        "MAT",  # Material
        "CMT",  # Characterization
        "DSC",  # Sample description
        "PRO",  # Property
        "SMT",  # Synthesis method
        "SPL"   # Symmetry/phase label
    ]
    
    # Initialize the NER agent for materials science
    print("="*70)
    print("           MATERIALS SCIENCE NER AGENT")
    print("="*70)
    print()
    
    print("Entity Tags:")
    entity_descriptions = {
        "APL": "Application - Use cases and applications",
        "MAT": "Material - Chemical compounds, elements, materials",
        "CMT": "Characterization - Analysis techniques and methods",
        "DSC": "Sample description - Sample composition and structure",
        "PRO": "Property - Physical and chemical properties",
        "SMT": "Synthesis method - Preparation and synthesis techniques",
        "SPL": "Symmetry/phase label - Crystal structure and phase information"
    }
    
    for tag, description in entity_descriptions.items():
        print(f"  {tag}: {description}")
    print()
    
    ner_agent = NERAgent(materials_entities)
    
    # Example 1: Provided sample text
    sample_text_1 = """Phololuminescence spectra of PbBr-based layered perovskites with an organic organic layer of napthalene-linked ammonium molecules."""
    
    print("Example 1 - Sample Text:")
    print(f'"{sample_text_1}"')
    print("\n" + "="*70)
    
    print("Extracting materials science entities...")
    result_1 = ner_agent.extract_entities(sample_text_1)
    
    print("\nExtracted Entities:")
    for entity_type, entities_list in result_1.items():
        if entities_list:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): {entities_list}")
        else:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): (No entities found)")
    
    print("\n" + "-"*70)
    print("JSON Output:")
    print(json.dumps(result_1, indent=2))
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Additional materials science text
    sample_text_2 = """The synthesis of TiO2 nanoparticles was carried out using sol-gel method at 450°C. 
    X-ray diffraction analysis revealed anatase phase with tetragonal symmetry. 
    The photocatalytic activity was measured under UV irradiation showing 85% degradation efficiency."""
    
    print("Example 2 - Additional Materials Science Text:")
    print(f'"{sample_text_2}"')
    print("\n" + "="*70)
    
    print("Extracting materials science entities...")
    result_2 = ner_agent.extract_entities(sample_text_2)
    
    print("\nExtracted Entities:")
    for entity_type, entities_list in result_2.items():
        if entities_list:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): {entities_list}")
        else:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): (No entities found)")
    
    print("\n" + "-"*70)
    print("JSON Output:")
    print(json.dumps(result_2, indent=2))
    
    print("\n" + "="*70 + "\n")
    
    # Example 3: Complex materials science text
    sample_text_3 = """Graphene oxide was functionalized with polyethyleneimine through covalent bonding 
    for water purification applications. The composite showed excellent adsorption capacity of 200 mg/g 
    for heavy metal ions. Scanning electron microscopy confirmed the layered structure."""
    
    print("Example 3 - Complex Materials Science Text:")
    print(f'"{sample_text_3}"')
    print("\n" + "="*70)
    
    print("Extracting materials science entities...")
    result_3 = ner_agent.extract_entities(sample_text_3)
    
    print("\nExtracted Entities:")
    for entity_type, entities_list in result_3.items():
        if entities_list:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): {entities_list}")
        else:
            print(f"{entity_type} ({entity_descriptions[entity_type]}): (No entities found)")
    
    print("\n" + "-"*70)
    print("JSON Output:")
    print(json.dumps(result_3, indent=2))
    
    print("\n" + "="*70)
    print("           BATCH PROCESSING EXAMPLE")
    print("="*70)
    
    # Batch processing example
    materials_texts = [
        sample_text_1,
        sample_text_2, 
        sample_text_3
    ]
    
    print("Processing multiple texts in batch...")
    batch_results = ner_agent.extract_entities_batch(materials_texts)
    
    for i, result in enumerate(batch_results, 1):
        print(f"\nText {i} Results:")
        print(json.dumps(result, indent=2))
        print("-" * 50)


def interactive_materials_science_ner():
    """Interactive materials science NER session"""
    
    materials_entities = ["APL", "MAT", "CMT", "DSC", "PRO", "SMT", "SPL"]
    
    entity_descriptions = {
        "APL": "Application - Use cases and applications",
        "MAT": "Material - Chemical compounds, elements, materials", 
        "CMT": "Characterization - Analysis techniques and methods",
        "DSC": "Sample description - Sample composition and structure",
        "PRO": "Property - Physical and chemical properties",
        "SMT": "Synthesis method - Preparation and synthesis techniques",
        "SPL": "Symmetry/phase label - Crystal structure and phase information"
    }
    
    print("="*70)
    print("    INTERACTIVE MATERIALS SCIENCE NER")
    print("="*70)
    print()
    print("Pre-configured entity tags for materials science:")
    for tag, description in entity_descriptions.items():
        print(f"  {tag}: {description}")
    print()
    
    ner_agent = NERAgent(materials_entities)
    
    while True:
        print("\n" + "="*70)
        print("Enter your materials science text:")
        print("(Press Ctrl+D (Unix/Mac) or Ctrl+Z (Windows) when finished)")
        print("-" * 40)
        
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        text = "\n".join(lines).strip()
        
        if not text:
            print("No text provided. Exiting...")
            break
        
        print(f"\nAnalyzing materials science text ({len(text)} characters)...")
        
        try:
            result = ner_agent.extract_entities(text)
            
            print("\n" + "="*70)
            print("              EXTRACTION RESULTS")
            print("="*70)
            
            found_entities = False
            for entity_type, entities_list in result.items():
                if entities_list:
                    found_entities = True
                    print(f"\n{entity_type} ({entity_descriptions[entity_type]}):")
                    for i, entity in enumerate(entities_list, 1):
                        print(f"  {i}. {entity}")
                else:
                    print(f"\n{entity_type} ({entity_descriptions[entity_type]}): (No entities found)")
            
            if not found_entities:
                print("No materials science entities found in the text.")
            
            print("\n" + "-" * 70)
            print("JSON Format:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
        
        print("\n" + "-" * 70)
        continue_choice = input("Analyze another materials science text? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("\nThank you for using the Materials Science NER Agent! 🔬")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_materials_science_ner()
    else:
        main()
