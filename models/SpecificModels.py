from models.LLModel import LLMModel, LLLAnswer, NonJSONLLMModel
from google import genai
import os
import logging
from pprint import pprint
import time
from datetime import datetime, timedelta

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                logger.warning("Limite de requisições por minuto atingido. Aguardando {:.2f} segundos.".format(wait_time))
            if tokens_last_minute + token_count > self.limit_tpm:
                if self.token_history:
                    earliest_token = min(t for t, count in self.token_history if t > one_minute_ago)
                    wait_time = (earliest_token + timedelta(minutes=1) - now).total_seconds()
                    wait_times.append(wait_time)
                    logger.warning("Limite de tokens por minuto atingido. Aguardando {:.2f} segundos.".format(wait_time))
                else:
                    wait_times.append(60)
                    logger.warning("Limite de tokens por minuto atingido. Aguardando 60 segundos.")
            if requests_today >= self.limit_rpd:
                earliest_daily = min(self.request_history)
                wait_time = (earliest_daily + timedelta(days=1) - now).total_seconds()
                wait_times.append(wait_time)
                logger.warning("Limite de requisições por dia atingido. Aguardando {:.2f} segundos.".format(wait_time))

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
    def wrapper(self, question, base_text, client):
        token_count = self.estimate_tokens(question, base_text)
        self.check_and_wait_limits(token_count)
        return func(self, question, base_text, client)
    return wrapper

# --- Classes de modelos com implementação de limite de API ---
class Gemini2_0Flash(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 2.0 Flash")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-2.0-flash"
        self.limit_rpm = 15         # Requisições por minuto
        self.limit_tpm = 1000000    # Tokens por minuto
        self.limit_rpd = 1500       # Requisições por dia

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini2_0FlashExperimental(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 2.0 Flash Experimental")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-2.0-flash-exp"
        self.limit_rpm = 10
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini2_0FlashLite(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 2.0 Flash Lite")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-2.0-flash-lite"
        self.limit_rpm = 30
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini2_0FlashThinkingExperimental(NonJSONLLMModel, RateLimitMixin):
    def __init__(self):
        NonJSONLLMModel.__init__(self, "Gemini 2.0 Flash Thinking Experimental")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-2.0-flash-thinking-exp-1219"
        self.limit_rpm = 10
        self.limit_tpm = 4000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini1_5Flash(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 1.5 Flash")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-1.5-flash"
        self.limit_rpm = 15
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini1_5Flash8B(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 1.5 Flash 8B")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-1.5-flash-8b"
        self.limit_rpm = 15
        self.limit_tpm = 1000000
        self.limit_rpd = 1500

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)

class Gemini1_5Pro(LLMModel, RateLimitMixin):
    def __init__(self):
        LLMModel.__init__(self, "Gemini 1.5 Pro")
        RateLimitMixin.__init__(self)
        self.model_name = "gemini-1.5-pro"
        self.limit_rpm = 2
        self.limit_tpm = 32000
        self.limit_rpd = 50

    @rate_limit_wait_decorator
    def generate_answer(self, question, base_text, client):
        return super().generate_answer(question, base_text, client, self.model_name)
