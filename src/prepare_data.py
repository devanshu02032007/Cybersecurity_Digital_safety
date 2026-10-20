import os
import csv
from pathlib import Path

# Define paths relative to the project root
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data/raw/urls/openphish_raw.txt"
OUT_FILE = ROOT / "data/raw/urls/urls.csv"

def clean_and_generate_csv():
    print(f"Reading raw data from: {RAW_FILE}")
    
    if not RAW_FILE.exists():
        print(f"Error: Raw file not found at {RAW_FILE}")
        return

    # Read raw lines using standard python file reading
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        phish_lines = [line.strip() for line in f if line.strip()]

    # Define a few baseline benign samples for classification balance
    benign_samples = [
        'https://www.google.com/',
        'https://github.com/',
        'https://www.wikipedia.org/',
        'https://www.python.org/',
        'https://stackoverflow.com/'
    ]

    # Write out cleanly using python's built-in csv writer
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['url', 'label'])
        
        # Write benign rows
        for b in benign_samples:
            writer.writerow([b, 'benign'])
            
        # Write phishing rows with safety parsing (stripping any stray commas)
        for p in phish_lines:
            clean_url = p.split(',')[0].strip()
            if clean_url:
                writer.writerow([clean_url, 'phishing'])

    print(f"Successfully generated clean CSV at: {OUT_FILE}")

if __name__ == "__main__":
    clean_and_generate_csv()
