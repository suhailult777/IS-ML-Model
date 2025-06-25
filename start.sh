#!/bin/bash

echo "==============================================="
echo "Medical Document Conversational AI"
echo "==============================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

echo "Starting Medical Document Conversational AI..."
echo

# Run the launch script
python3 launch.py

echo
echo "Session ended. Goodbye!"
