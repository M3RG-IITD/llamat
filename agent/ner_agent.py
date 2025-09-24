"""
Named Entity Recognition Agent using local LLM model
"""

import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NERConfig:
    """Configuration for NER agent"""
    model_endpoint: str = "http://localhost:8000/v1/completions"
    model_name: str = "models--m3rg-iitd--llamat-2-chat"
    temperature: float = 0.1
    max_tokens: int = 512


class NERAgent:
    """
    Named Entity Recognition Agent that extracts specified entities from text
    """
    
    def __init__(self, entities: List[str], config: Optional[NERConfig] = None):
        """
        Initialize NER Agent
        
        Args:
            entities: List of entity types to extract (e.g., ['PERSON', 'ORG', 'LOCATION'])
            config: Configuration for the model endpoint
        """
        self.entities = entities
        self.config = config or NERConfig()
        
    def _create_prompt(self, text: str) -> str:
        """Create the prompt for entity extraction"""
        entities_str = ", ".join(self.entities)
        
        # Create the JSON structure template
        json_template = "{\n"
        for i, entity in enumerate(self.entities):
            json_template += f'    "{entity}": []'
            if i < len(self.entities) - 1:
                json_template += ","
            json_template += "\n"
        json_template += "}"
        
        # Check if this looks like materials science content
        is_materials_science = any(tag in self.entities for tag in ["APL", "MAT", "CMT", "DSC", "PRO", "SMT", "SPL"])
        
        if is_materials_science:
            prompt = f"""You are a materials science expert. Extract the following entity types from the text: {entities_str}

Text: "{text}"

Entity definitions:
- APL (Application): Use cases, applications, purposes
- MAT (Material): Chemical compounds, elements, materials, molecules
- CMT (Characterization): Analysis techniques, measurement methods, characterization tools
- DSC (Sample description): Sample composition, structure, morphology
- PRO (Property): Physical properties, chemical properties, performance metrics
- SMT (Synthesis method): Preparation methods, synthesis techniques, processing
- SPL (Symmetry/phase label): Crystal structures, phases, symmetry groups

Return ONLY a single JSON object in this exact format:
{json_template}

Rules:
1. Return only the JSON object, no other text
2. If no entities of a type are found, use an empty list []
3. Extract complete entity names and phrases
4. Do not include empty strings in lists
5. For materials science: include chemical formulas, technique names, property values

JSON:"""
        else:
            prompt = f"""Extract the following entities from the text: {entities_str}

Text: "{text}"

Return ONLY a single JSON object in this exact format:
{json_template}

Rules:
1. Return only the JSON object, no other text
2. If no entities of a type are found, use an empty list []
3. Extract complete entity names
4. Do not include empty strings in lists

JSON:"""
        
        return prompt
    
    def _call_model(self, prompt: str) -> str:
        """Call the local model API"""
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = requests.post(
                self.config.model_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling model API: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Error parsing model response: {e}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from the given text
        
        Args:
            text: Input text paragraph
            
        Returns:
            Dictionary with entity types as keys and lists of extracted entities as values
        """
        prompt = self._create_prompt(text)
        
        try:
            # Get response from model
            model_response = self._call_model(prompt)
            
            # Parse JSON response
            try:
                # Clean the response to extract JSON
                if "```json" in model_response:
                    json_start = model_response.find("```json") + 7
                    json_end = model_response.find("```", json_start)
                    json_str = model_response[json_start:json_end].strip()
                elif "```" in model_response:
                    json_start = model_response.find("```") + 3
                    json_end = model_response.find("```", json_start)
                    json_str = model_response[json_start:json_end].strip()
                else:
                    # Try to find the first complete JSON object in the response
                    start_idx = model_response.find('{')
                    if start_idx != -1:
                        # Find the matching closing brace
                        brace_count = 0
                        end_idx = start_idx
                        for i, char in enumerate(model_response[start_idx:], start_idx):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        json_str = model_response[start_idx:end_idx]
                    else:
                        raise ValueError("No JSON object found in response")
                
                extracted_entities = json.loads(json_str)
                
                # Ensure all entity types are present with empty lists if not found
                result = {}
                for entity_type in self.entities:
                    entities_list = extracted_entities.get(entity_type, [])
                    # Filter out empty strings
                    result[entity_type] = [entity for entity in entities_list if entity and entity.strip()]
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON response: {e}")
                print(f"Raw response: {model_response}")
                
                # Fallback: return empty lists for all entity types
                return {entity_type: [] for entity_type in self.entities}
                
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            # Return empty result on error
            return {entity_type: [] for entity_type in self.entities}
    
    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts
        
        Args:
            texts: List of input text paragraphs
            
        Returns:
            List of dictionaries with extracted entities for each text
        """
        results = []
        for text in texts:
            result = self.extract_entities(text)
            results.append(result)
        return results


def main():
    """Example usage of the NER Agent"""
    
    # Define entities to extract
    entities = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"]
    
    # Initialize the agent
    ner_agent = NERAgent(entities)
    
    # Example text
    sample_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California on April 1, 1976. 
    The company is headquartered in Cupertino and has offices worldwide. In 2023, Apple reported revenue of $394.3 billion.
    Tim Cook is the current CEO of Apple.
    """
    
    print("Sample text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Extract entities
    print("Extracting entities...")
    result = ner_agent.extract_entities(sample_text)
    
    print("Extracted entities:")
    for entity_type, entities_list in result.items():
        print(f"{entity_type}: {entities_list}")
    
    print("\n" + "="*50 + "\n")
    
    # Pretty print as JSON
    print("Result as JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
