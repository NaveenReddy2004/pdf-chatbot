import requests
import os
from dotenv import load_dotenv
load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_API_TOKEN")

def get_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})

    if response.status_code != 200:
        print("Embedding API error:", response.text)
        return []

    data = response.json()

    # Sanity check
    if isinstance(data, list) and isinstance(data[0], list):
        return data  
    else:
        print("Unexpected embedding format:", data)
        return []
