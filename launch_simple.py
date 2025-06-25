#!/usr/bin/env python3
"""
Launch script for PDF Conversational AI
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import sentence_transformers
        import faiss
        import numpy as np
        import pickle
        import json
        logger.info("✓ All basic requirements are available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing requirement: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("PDF CONVERSATIONAL AI LAUNCHER")
    print("=" * 60)
    
    print("Checking system requirements...")
    if not check_requirements():
        print("Please install missing requirements with: pip install -r requirements.txt")
        return 1
    
    print("✓ System ready!")
    print("\nChoose an option:")
    print("1. Start Simple Chat Interface (Recommended)")
    print("2. Start Advanced Training Interface")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                print("\nStarting Simple Chat Interface...")
                print("This uses pre-processed PDF content for fast responses.\n")
                try:
                    from simple_chat import SimplePDFChat
                    chat = SimplePDFChat()
                    chat.chat_loop()
                except Exception as e:
                    logger.error(f"Error starting simple chat: {e}")
                    return 1
                break
                
            elif choice == '2':
                print("\nStarting Advanced Training Interface...")
                print("This will train a neural model (may take time).\n")
                try:
                    subprocess.run([sys.executable, "conversational_ai.py"], check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error starting advanced interface: {e}")
                    return 1
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                break
                
            elif choice == '3':
                print("Goodbye!")
                return 0
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
