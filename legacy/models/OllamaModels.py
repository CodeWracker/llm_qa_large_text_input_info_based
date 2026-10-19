from models.LLMModel import  LLMAnswer, LLMModel, NonJSONLLMModel
import os
import logging
from pprint import pprint
import time
from datetime import datetime, timedelta
import json
import requests


class OllamaModel(NonJSONLLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
    def query_model(self, prompt: str, model_name_query: str) -> LLMAnswer:
        payload = {
            "model": model_name_query,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        while True:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    headers=headers,
                    data=json.dumps(payload),
                )
                response.raise_for_status()
                content = response.json()["response"]
                logging.debug(f"Model {model_name_query} answered: {content}")
                answer = self.convert_answer_to_LLMAnswer(content)
                return answer
            except ValueError as e:
                logging.error(f"OllamaModels.py - Failed to convert answer to LLMAnswer: {e}")
                prompt = prompt + f"\nYou have already answered this question, but your answer was not in the correct format. Please answer again, but this time make sure to follow the instructions and return a valid JSON. Previous error: {e}"
        
            except Exception as e:
                logging.error(f"OllamaModels.py - Error querying model {model_name_query}: {e}")
                

class DeepSeekR1_1_5b(OllamaModel):
    def __init__(self):
        super().__init__("Ollama DeepSeek R1 1.5b")
        self.model_name = "deepseek-r1:1.5b"
    
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)
    
class Llama32_1b(OllamaModel):
    def __init__(self):
        super().__init__("Ollama Llama 3 1b")
        self.model_name = "llama3.2:1b"
    
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemma3_1b(OllamaModel):
    def __init__(self):
        super().__init__("Ollama Gemma 3 1b")
        self.model_name = "gemma3:1b"
    
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)