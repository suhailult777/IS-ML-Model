#!/usr/bin/env python3
"""
Find exact treatment text
"""

import json

# Load the PDF content directly
with open('data/processed/pdf_content.json', 'r') as f:
    data = json.load(f)
    content = data['text']

# Find where "Treatment for AKI" appears
treatment_start = content.find('Treatment for AKI')
if treatment_start != -1:
    # Extract a large chunk around it
    treatment_chunk = content[treatment_start:treatment_start+1000]
    print("Treatment section found:")
    print(repr(treatment_chunk))
    print("\n" + "="*50)
    print("Cleaned version:")
    
    import re
    clean_chunk = re.sub(r'\s+', ' ', treatment_chunk)
    print(clean_chunk)
    
    # Look for dialysis specifically
    dialysis_start = content.find('dialysis')
    if dialysis_start != -1:
        dialysis_chunk = content[dialysis_start-200:dialysis_start+300]
        print("\n" + "="*50)
        print("Dialysis section:")
        print(repr(dialysis_chunk))
