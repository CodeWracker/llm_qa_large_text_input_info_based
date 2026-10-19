from models.LLMModel import  LLMAnswer, LLMModel, NonJSONLLMModel
import os
import logging
from pprint import pprint
import time
from datetime import datetime, timedelta

from google import genai



# --- Mixin e Decorator para controle de limites ---
class RateLimitMixin:
    def __init__(self):
        self.request_history = []  # Armazena os timestamps das requisições
        self.token_history = []    # Armazena tuplas (timestamp, quantidade de tokens)

    def estimate_tokens(self, *texts):
        """Estimativa simples contando palavras (substitua por um tokenizador real, se necessário)."""
        return sum(len(text.split()) for text in texts if text)

    def check_and_wait_limits(self, token_count):
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_day_ago = now - timedelta(days=1)

        # Remove entradas antigas dos históricos
        self.request_history = [t for t in self.request_history if t > one_day_ago]
        self.token_history = [(t, count) for t, count in self.token_history if t > one_minute_ago]

        tokens_last_minute = sum(count for t, count in self.token_history)
        requests_last_minute = len([t for t in self.request_history if t > one_minute_ago])
        requests_today = len(self.request_history)

        # Enquanto algum dos limites for ultrapassado, aguarda
        while (requests_last_minute >= self.limit_rpm or
               tokens_last_minute + token_count > self.limit_tpm or
               requests_today >= self.limit_rpd):
            wait_times = []
            if requests_last_minute >= self.limit_rpm:
                earliest = min(t for t in self.request_history if t > one_minute_ago)
                wait_time = (earliest + timedelta(minutes=1) - now).total_seconds()
                wait_times.append(wait_time)
                logging.warning("Limite de requisições por minuto atingido. Aguardando {:.2f} segundos.".format(wait_time))
            if tokens_last_minute + token_count > self.limit_tpm:
                if self.token_history:
                    earliest_token = min(t for t, count in self.token_history if t > one_minute_ago)
                    wait_time = (earliest_token + timedelta(minutes=1) - now).total_seconds()
                    wait_times.append(wait_time)
                    logging.warning("Limite de tokens por minuto atingido. Aguardando {:.2f} segundos.".format(wait_time))
                else:
                    wait_times.append(60)
                    logging.warning("Limite de tokens por minuto atingido. Aguardando 60 segundos.")
            if requests_today >= self.limit_rpd:
                earliest_daily = min(self.request_history)
                wait_time = (earliest_daily + timedelta(days=1) - now).total_seconds()
                wait_times.append(wait_time)
                logging.warning("Limite de requisições por dia atingido. Aguardando {:.2f} segundos.".format(wait_time))

            # Aguarda o tempo máximo necessário dentre os limites
            sleep_time = max(wait_times)
            if sleep_time < 0:
                sleep_time = 0
            time.sleep(sleep_time)
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            one_day_ago = now - timedelta(days=1)
            self.request_history = [t for t in self.request_history if t > one_day_ago]
            self.token_history = [(t, count) for t, count in self.token_history if t > one_minute_ago]
            tokens_last_minute = sum(count for t, count in self.token_history)
            requests_last_minute = len([t for t in self.request_history if t > one_minute_ago])
            requests_today = len(self.request_history)

        # Registra a requisição atual
        self.request_history.append(now)
        self.token_history.append((now, token_count))

def rate_limit_wait_decorator(func):
    def wrapper(self, question, base_text):
        token_count = self.estimate_tokens(question, base_text)
        self.check_and_wait_limits(token_count)
        return func(self, question, base_text)
    return wrapper


GeminiClient = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


class GoogleLLMModel(RateLimitMixin):
    def __init__(self):
        super().__init__()
        self.client = GeminiClient
        
    def __del__(self):

        self.client.__del__ = lambda self: None
        self.client = None

class GoogleLLMModelNonJSON(NonJSONLLMModel, GoogleLLMModel):
    def __init__(self, model_name):
        NonJSONLLMModel.__init__(self,model_name)
        GoogleLLMModel.__init__(self)
        
        
    def query_model(self, prompt,model_name_query):
        while True:
            try:
                response = self.client.models.generate_content(
                    model=model_name_query,
                    contents=prompt
                )
            # exception for RESOURCE_EXHAUSTED
            except Exception as e:
                # 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-1.5-pro'}, 'quotaValue': '50'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '44s'}]}}
                # se e contem 'RESOURCE_EXHAUSTED'
                if "RESOURCE_EXHAUSTED" in str(e):
                    logging.error(f"GoogleModels.py - Resource exhausted: {e}")
                    logging.warning("Waiting for 1minute before retrying...")
                    time.sleep(60)
                # checka se é keyboard e se sim, espera 30 segundos
                elif "KeyboardInterrupt" in str(e):
                    raise KeyboardInterrupt
                else:
                    logging.error(f"GoogleModels.py - An error occurred: {e}")
                    logging.warning("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                continue
            # logging.info(f"Response: {response}")
            
            try:
                model_answer = self.convert_answer_to_LLMAnswer(response.candidates[0].content.parts[0].text)
                return model_answer
            except ValueError as e:
                logging.error(f"GoogleModels.py - Failed to convert answer to LLMAnswer: {e}")
                prompt = prompt + f"\nYou have already answered this question, but your answer was not in the correct format. Please answer again, but this time make sure to follow the instructions and return a valid JSON. Previous error: {e}"
        



class GoogleLLMModelJSON(LLMModel, GoogleLLMModel):
    
    def __init__(self, model_name):
        LLMModel.__init__(self, model_name)
        GoogleLLMModel.__init__(self)
        
        
    
        
    def query_model(self, prompt, model_name_query):
        """
        Send the prompt to a model until a valid JSON answer is returned.
        """
        
        while True:
            try:
                response = self.client.models.generate_content(
                    model=model_name_query,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': LLMAnswer
                    }
                )
                break
            except Exception as e:
                # 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-1.5-pro'}, 'quotaValue': '50'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '44s'}]}}
                # se e contem 'RESOURCE_EXHAUSTED'
                if "RESOURCE_EXHAUSTED" in str(e):
                    logging.error(f"GoogleModels.py - Resource exhausted: {e}")
                    logging.warning("Waiting for 1minute before retrying...")
                    time.sleep(60)
                # checka se é keyboard e se sim, espera 30 segundos
                elif "KeyboardInterrupt" in str(e):
                    raise KeyboardInterrupt
                else:
                    logging.error(f"GoogleModels.py - An error occurred: {e}")
                    logging.warning("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                continue
        return response.parsed

# --- Classes de modelos com implementação de limite de API ---
class Gemini2_0Flash(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 2.0 Flash")
        self.model_name = "gemini-2.0-flash"
        self.limit_rpm = 15         # Requisições por minuto
        self.limit_tpm = 1000000    # Tokens por minuto
        self.limit_rpd = 1500       # Requisições por dia

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini2_0FlashExperimental(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 2.0 Flash Experimental")
        self.model_name = "gemini-2.0-flash-exp"
        self.limit_rpm = 10
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini2_0FlashLite(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 2.0 Flash Lite")
        self.model_name = "gemini-2.0-flash-lite"
        self.limit_rpm = 30
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini2_0FlashThinkingExperimental(GoogleLLMModelNonJSON):
    def __init__(self):
        GoogleLLMModelNonJSON.__init__(self, "Gemini 2.0 Flash Thinking Experimental")
        self.model_name = "gemini-2.0-flash-thinking-exp-1219"
        self.limit_rpm = 10
        self.limit_tpm = 4000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini1_5Flash(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 1.5 Flash")
        self.model_name = "gemini-1.5-flash"
        self.limit_rpm = 15
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini1_5Flash8B(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 1.5 Flash 8B")
        self.model_name = "gemini-1.5-flash-8b"
        self.limit_rpm = 15
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)

class Gemini1_5Pro(GoogleLLMModelJSON):
    def __init__(self):
        GoogleLLMModelJSON.__init__(self, "Gemini 1.5 Pro")
        self.model_name = "gemini-1.5-pro"
        self.limit_rpm = 2
        self.limit_tpm = 32000
        self.limit_rpd = 50

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text):
        return super().generate_answer(question, base_text, self.model_name)
