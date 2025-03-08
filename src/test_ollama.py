#!/usr/bin/env python3
"""
Test script for Ollama integration with AI Hedge Fund
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from llm.models import ModelProvider, get_model

# Load environment variables
load_dotenv()

def test_ollama():
    """Test Ollama integration"""
    print("Testing Ollama integration...")
    
    # Get the Ollama model
    model_name = "mixtral:8x7b"  # Change this to any model you have pulled in Ollama
    
    try:
        # Initialize the model
        llm = get_model(model_name, ModelProvider.OLLAMA)
        
        if not llm:
            print("❌ Failed to initialize Ollama model")
            return False
        
        # Test a simple query
        response = llm.invoke([
            HumanMessage(content="What is the capital of France?")
        ])
        
        print("\n✅ Ollama model initialized successfully!")
        print(f"\nTest query response:\n{response.content}")
        return True
    
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

if __name__ == "__main__":
    # Test Ollama
    success = test_ollama()
    
    if success:
        print("\n✅ Ollama integration test completed successfully!")
    else:
        print("\n❌ Ollama integration test failed.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is installed and running (https://ollama.com)")
        print("2. Check that you have pulled the model you're trying to use (e.g., 'ollama pull llama3')")
        print("3. Verify that Ollama is accessible at http://localhost:11434 or set OLLAMA_BASE_URL in your .env file")
