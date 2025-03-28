from models.LLModel import LLMModel,LLLAnswer

from google import genai
import os
# genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
import logging
from pprint import pprint
# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
Modelo	RPM	TPM	RPD
Gemini 2.5 Pro Experimental	2	1.000.000	50
Gemini 2.0 Flash	15	1.000.000	1.500
Gemini 2.0 Flash Experimental	10	1.000.000	1.500
Gemini 2.0 Flash-Lite	30	1.000.000	1.500
Gemini 2.0 Flash Thinking Experimental 01-21	10	4.000.000	1.500
Gemini 1.5 Flash	15	1.000.000	1.500
Gemini 1.5 Flash-8B	15	1.000.000	1.500
Gemini 1.5 Pro	2	32.000	50
Imagem 3	--	--	--
O Gemma 3	30	15.000	14.400
Gemini Embedding Experimental 03-07	5	--	100
"""


class Gemini2_5ProExperimental(LLMModel):
    def __init__(self):
        super().__init__("Gemini 2.5 Pro Experimental")
        self.model_name = "gemini-2.5-pro-experimental"
        
        self.limit_rpm = 2 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 50 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)

class Gemini2_0Flash(LLMModel):
    def __init__(self):
        super().__init__("Gemini 2.0 Flash")
        self.model_name = "gemini-2.0-flash"
        
        self.limit_rpm = 15 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class Gemini2_0FlashExperimental(LLMModel):
    def __init__(self):
        super().__init__("Gemini 2.0 Flash Experimental")
        self.model_name = "gemini-2.0-flash-experimental"
        
        self.limit_rpm = 10 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class Gemini2_0FlashLite(LLMModel):
    def __init__(self):
        super().__init__("Gemini 2.0 Flash Lite")
        self.model_name = "gemini-2.0-flash-lite"
        
        self.limit_rpm = 30 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class Gemini2_0FlashThinkingExperimental(LLMModel):
    def __init__(self):
        super().__init__("Gemini 2.0 Flash Thinking Experimental")
        self.model_name = "gemini-2.0-flash-thinking-experimental"
        
        self.limit_rpm = 10 # Limite de requisições por minuto
        self.limit_tpm = 4000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class Gemini1_5Flash(LLMModel):
    def __init__(self):
        super().__init__("Gemini 1.5 Flash")
        self.model_name = "gemini-1.5-flash"
        
        self.limit_rpm = 15 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        

    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)

class Gemini1_5Flash8B(LLMModel):
    def __init__(self):
        super().__init__("Gemini 1.5 Flash 8B")
        self.model_name = "gemini-1.5-flash-8b"
        
        self.limit_rpm = 15 # Limite de requisições por minuto
        self.limit_tpm = 1000000 # Limite de tokens por minuto
        self.limit_rpd = 1500 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class Gemini1_5Pro(LLMModel):
    def __init__(self):
        super().__init__("Gemini 1.5 Pro")
        self.model_name = "gemini-1.5-pro"
        
        self.limit_rpm = 2 # Limite de requisições por minuto
        self.limit_tpm = 32000 # Limite de tokens por minuto
        self.limit_rpd = 50 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)
    
class GeminiGemma3(LLMModel):
    def __init__(self):
        super().__init__("Gemini Gemma 3")
        self.model_name = "gemini-gemma-3"
        
        self.limit_rpm = 30 # Limite de requisições por minuto
        self.limit_tpm = 15000 # Limite de tokens por minuto
        self.limit_rpd = 14400 # Limite de requisições por dia
        
    def generate_answer(self, question, base_text, client):
        return self.super().generate_answer(question, base_text, client, self.model_name)