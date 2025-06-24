import os
import requests
from langchain.llms.base import LLM
from typing import List
from dotenv import load_dotenv
load_dotenv()

class GroqLLM(LLM):
    model = "llama3-70b-8192"
    groq_api_key = os.getenv("GROQ_API_KEY")

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "groq_llm"
