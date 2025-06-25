#!/usr/bin/env python3
"""
Simple test for treatment question
"""

import sys
import os
import json

# Load the PDF content directly
try:
    with open('data/processed/pdf_content.json', 'r') as f:
        data = json.load(f)
        content = data['text']
        
    print("Sample content length:", len(content))
    print("\nLooking for treatment information...")
    
    # Find treatment section
    start_idx = content.find('Treatment for AKI')
    if start_idx != -1:
        treatment_section = content[start_idx:start_idx+500]
        print("Found treatment section:")
        print(treatment_section)
    else:
        print("Treatment section not found")
        
    # Look for dialysis mention
    dialysis_idx = content.find('dialysis')
    if dialysis_idx != -1:
        dialysis_section = content[dialysis_idx-100:dialysis_idx+200]
        print("\nFound dialysis mention:")
        print(dialysis_section)
        
except Exception as e:
    print("Error:", e)
