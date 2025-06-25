#!/usr/bin/env python3
"""
Startup Script for Medical Document Conversational AI
Simple launcher with system checks and setup
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'transformers',
        'sentence-transformers',
        'numpy',
        'scikit-learn',
        'faiss-cpu',  # or 'faiss-gpu' if available
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Special handling for faiss
            if package.startswith('faiss'):
                try:
                    import faiss
                    print(f"✅ {package} is installed")
                except ImportError:
                    missing_packages.append(package)
                    print(f"❌ {package} is missing")
            else:
                importlib.import_module(package)
                print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        print("📦 Installing requirements from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("⚠️  requirements.txt not found")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/processed/pdf_content.txt",
        "data/processed/qa_pairs.json",
        "data/processed/knowledge_base.json"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} is missing")
    
    return missing_files

def setup_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory {directory} ready")

def run_system_check():
    """Run complete system check"""
    print("🔍 Running system check...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print()
    
    # Check required packages
    print("📋 Checking required packages...")
    missing_packages = check_required_packages()
    
    if missing_packages:
        print(f"\\n⚠️  Missing packages: {', '.join(missing_packages)}")
        
        # Try to install from requirements.txt
        if not install_requirements():
            print("\\n❌ Please install missing packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    print()
    
    # Setup directories
    print("📁 Setting up directories...")
    setup_directories()
    
    print()
    
    # Check data files
    print("📄 Checking data files...")
    missing_files = check_data_files()
    
    if missing_files:
        print(f"\\n⚠️  Missing data files: {', '.join(missing_files)}")
        print("The system will try to process available data or train from scratch.")
    
    print("\\n✅ System check completed!")
    return True

def launch_conversational_ai():
    """Launch the conversational AI system"""
    print("\\n🚀 Launching Medical Document Conversational AI...")
    print("=" * 50)
    
    try:
        # Import and run the conversational AI
        from conversational_ai import main
        return main()
    except ImportError as e:
        print(f"❌ Failed to import conversational AI: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to launch conversational AI: {e}")
        return 1

def main():
    """Main startup function"""
    print("🏥 Medical Document Conversational AI - Startup Script")
    print("=" * 60)
    
    # Run system check
    if not run_system_check():
        print("\\n❌ System check failed. Please fix the issues above and try again.")
        return 1
    
    # Launch the system
    return launch_conversational_ai()

if __name__ == "__main__":
    sys.exit(main())
