"""
Materials Science Chat Agent
Interactive chat interface for materials science questions using local LLM
"""

import json
import os
import pickle
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class ChatConfig:
    """Configuration for chat agent"""
    model_endpoint: str = "http://localhost:8000/v1/completions"
    # model_name: str = ".cache/huggingface/hub/models--m3rg-iitd--llamat-2-chat/snapshots/2b67f6910c90d34e04ef5cb39ae0e5d7ae2e1259"
    model_name: str = "/home/mzaki4/scr16_mshiel10/mzaki4/mzaki4/cache/models--m3rg-iitd--llamat-2-chat/snapshots/2b67f6910c90d34e04ef5cb39ae0e5d7ae2e1259" 
    temperature: float = 0.7
    max_tokens: int = 512
    max_context_tokens: int = 3000  # Token limit for context management
    cache_dir: str = "cache"
    system_prompt: str = """You are a materials science expert assistant. You have deep knowledge of:
- Materials chemistry and physics
- Synthesis and characterization techniques
- Crystal structures and phase diagrams
- Electronic, optical, and mechanical properties
- Applications in various industries
- Recent advances in materials science

Provide concise, accurate, and helpful answers to materials science questions. Be direct and to-the-point while maintaining scientific accuracy. Use appropriate terminology but avoid unnecessary verbosity."""


class ChatAgent:
    """
    Materials Science Chat Agent with conversation history management
    """
    
    def __init__(self, config: Optional[ChatConfig] = None):
        """Initialize the chat agent"""
        self.config = config or ChatConfig()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_tokens = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": self.config.system_prompt
        })
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (approximate: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _get_conversation_hash(self) -> str:
        """Generate a hash for the current conversation"""
        conversation_str = json.dumps(self.conversation_history, sort_keys=True)
        return hashlib.md5(conversation_str.encode()).hexdigest()[:12]
    
    def _save_conversation_cache(self):
        """Save current conversation to cache"""
        cache_file = os.path.join(
            self.config.cache_dir, 
            f"chat_cache_{self._get_conversation_hash()}.pkl"
        )
        
        cache_data = {
            "conversation_history": self.conversation_history,
            "current_tokens": self.current_tokens,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_conversation_cache(self, cache_hash: str) -> bool:
        """Load conversation from cache"""
        cache_file = os.path.join(
            self.config.cache_dir, 
            f"chat_cache_{cache_hash}.pkl"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.conversation_history = cache_data["conversation_history"]
                self.current_tokens = cache_data["current_tokens"]
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
                return False
        return False
    
    def _manage_context(self):
        """Manage conversation context to stay within token limits"""
        if self.current_tokens <= self.config.max_context_tokens:
            return
        
        # Keep system message and recent conversation
        system_msg = self.conversation_history[0]
        recent_messages = []
        current_tokens = self._estimate_tokens(system_msg["content"])
        
        # Keep as many recent messages as possible within limit
        for message in reversed(self.conversation_history[1:]):
            message_tokens = self._estimate_tokens(message["content"])
            if current_tokens + message_tokens <= self.config.max_context_tokens:
                recent_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                break
        
        # Rebuild conversation history
        self.conversation_history = [system_msg] + recent_messages
        self.current_tokens = current_tokens
        
        # Save to cache when context is managed
        self._save_conversation_cache()
    
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """Call the local model API"""
        # Convert messages to a single prompt for completions endpoint
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        
        # Improved prompt to get concise, direct responses
        prompt += """Assistant:"""
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": min(self.config.max_tokens, 200),  # Limit response length
            "stop": ["\n\nUser:", "\n\nAssistant:", "User:", "Assistant:"]
        }
        
        try:
            response = requests.post(
                self.config.model_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result["choices"][0]["text"].strip()
            
            # Clean up the response to remove any unwanted tags
            cleaned_response = self._clean_response(response_text)
            
            return cleaned_response
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling model API: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Error parsing model response: {e}")
    
    def _clean_response(self, response: str) -> str:
        """Clean the model response to remove unwanted tags and formatting"""
        if not response:
            return response
        
        # Remove any remaining User: or Assistant: tags at the beginning
        response = response.strip()
        
        # Remove leading "Assistant:" if present
        if response.startswith('Assistant:'):
            response = response[10:].strip()
        
        # Remove leading "User:" if present (shouldn't happen but just in case)
        if response.startswith('User:'):
            response = response[5:].strip()
        
        # Remove [Response] tags
        response = response.replace('[Response]', '').strip()
        
        # Split into lines and clean
        lines = response.split('\n')
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
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return assistant response
        
        Args:
            user_message: The user's question/message
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Update token count
        self.current_tokens += self._estimate_tokens(user_message)
        
        # Manage context if needed
        self._manage_context()
        
        try:
            # Get response from model
            response = self._call_model(self.conversation_history)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Update token count
            self.current_tokens += self._estimate_tokens(response)
            
            return response
            
        except Exception as e:
            # Remove user message on error
            self.conversation_history.pop()
            self.current_tokens -= self._estimate_tokens(user_message)
            raise e
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = [{
            "role": "system",
            "content": self.config.system_prompt
        }]
        self.current_tokens = self._estimate_tokens(self.config.system_prompt)
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            "total_messages": len(self.conversation_history) - 1,  # Exclude system message
            "current_tokens": self.current_tokens,
            "cache_hash": self._get_conversation_hash(),
            "last_message": self.conversation_history[-1] if len(self.conversation_history) > 1 else None
        }
    
    def export_conversation(self) -> List[Dict[str, str]]:
        """Export conversation history (excluding system message)"""
        return self.conversation_history[1:]


def main():
    """Simple command-line chat interface"""
    print("="*60)
    print("        MATERIALS SCIENCE CHAT AGENT")
    print("="*60)
    print("Ask me anything about materials science!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Type 'summary' to see conversation info.")
    print()
    
    agent = ChatAgent()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                agent.clear_history()
                print("Conversation cleared. Starting fresh!")
                continue
            
            if user_input.lower() == 'summary':
                summary = agent.get_conversation_summary()
                print(f"Messages: {summary['total_messages']}, Tokens: {summary['current_tokens']}")
                continue
            
            if not user_input:
                continue
            
            print("Assistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
