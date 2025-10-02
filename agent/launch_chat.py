#!/usr/bin/env python3
"""
Launch script for Materials Science Chat Interface
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def check_model_availability():
    """Check if the local model is available"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Model server is running")
            return True
        else:
            print("❌ Model server is not responding correctly")
            return False
    except requests.exceptions.RequestException:
        print("❌ Model server is not available at http://localhost:8000")
        print("Please make sure your llamat-2 model is running")
        return False


def main():
    """Main launch function"""
    print("🧪 Materials Science Chat Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model availability
    if not check_model_availability():
        print("\n⚠️  Warning: Model server is not available.")
        print("The chat interface will still launch, but you may encounter errors.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Launch Streamlit app
    print("\n🚀 Launching Streamlit chat interface...")
    print("The app will open in your default browser.")
    print("Press Ctrl+C to stop the server.")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(script_dir / "streamlit_chat.py"),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped.")
    except Exception as e:
        print(f"❌ Error launching chat interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
