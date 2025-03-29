from pydantic import BaseModel

import logging
from google import genai
import json
import os
import re
import time
class LLLAnswer(BaseModel):
    question: str
    unanswerable: bool
    answer: str
    




class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        
    # virtual method to be implemented by subclasses
    def generate_answer(self, question, base_text, client, model_name):
        prompt = self.generate_prompt(question, base_text)
        while True:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': LLLAnswer
                    }
                )
                break
            except Exception as e:
                # 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-1.5-pro'}, 'quotaValue': '50'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '44s'}]}}
                # se e contem 'RESOURCE_EXHAUSTED'
                if "RESOURCE_EXHAUSTED" in str(e):
                    logging.error(f"Resource exhausted: {e}")
                    logging.warning("Waiting for 1minute before retrying...")
                    time.sleep(60)
                # checka se é keyboard e se sim, espera 30 segundos
                elif "KeyboardInterrupt" in str(e):
                    raise KeyboardInterrupt
                else:
                    logging.error(f"An error occurred: {e}")
                    logging.warning("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                continue
        model_answer = response.parsed
        
        
        
        return model_answer
        

    def generate_prompt(self, question, base_text):
        
        json_example = f'"""json\n{{"question": "{question}", "unanswerable": false, "answer": "YOUR ANSWER HERE"}}\n"""'
        prompt = f"""
        ATENÇÃO: SUA RESPOSTA DEVE SER SOMENTE UM JSON! NÃO INCLUA TEXTO EXPLICATIVO, SOMENTE O JSON. SE UM CAMPO NÃO PUDER SER PREENCHIDO E NÃO TE DEI INSCTRUÇÕES DO QUE COLOCAR, USE "N/A". A ESTRUTURA DEVE SER EXATAMENTE COMO INDICADA ABAIXO.
        **NÃO ADICIONE TEXTO EXPLICATIVO OU EXTRA**. A resposta deve ser limitada apenas à representação do JSON, seguindo a estrutura especificada e em Inglês.
        
        ```json
        {json_example}
        ```
        
        **Instruções**
        - **MANTENHA O FORMATO DO JSON**: A resposta deve ser um JSON válido, com os campos "question" (string), "unanswerable" (boolean) e "answer" (string).
        - Responda com "N/A" se não souber a resposta ou se o campo não puder ser preenchido.
        - **Não adicione texto explicativo**: A resposta deve ser limitada apenas à representação do JSON, seguindo a estrutura especificada e em Inglês.
        - Não importa em qual lingua a pergunta foi feita ou qual lingua o texto base está, a resposta deve ser sempre em inglês.
        - As keys do json devem ser entre aspas duplas.
        - Responda a pergunta: "{question}"
        
        ----
        
        **Texto Base**: 
        {base_text}
        
        ----
        
        **Pergunta**: {question}
        
        """
        
        return prompt
    
    
class NonJSONLLMModel(LLMModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def convert_answer_to_LLMAnswer(self, answer):
        # First checks if the answer is a valid json and then maps it to the LLMAnswer class
        # logging.info(f"Converting answer to LLMAnswer: {answer}")
        
        # o texto esta no seguinte formato ```json\n{json}\n``` e o json é o que queremos (junto com os {})
        
        answer_parsed = ""
        for line in answer.split("\n")[1:-1]:
            answer_parsed += line.strip()
        answer = answer_parsed
            
        # logging.info(f"Extracted JSON: {answer}")
        
        parsed_answer = json.loads(answer)
        # logging.info(f"Parsed answer: {parsed_answer}")
        answer = LLLAnswer(
            question=parsed_answer.get("question", None),
            unanswerable=parsed_answer.get("unanswerable", False),
            answer=parsed_answer.get("answer", None)
        )
        
        return answer
        
    # virtual method to be implemented by subclasses
    def generate_answer(self, question, base_text, client, model_name):
        prompt = self.generate_prompt(question, base_text)

        while True:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            # exception for RESOURCE_EXHAUSTED
            except Exception as e:
                # 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-1.5-pro'}, 'quotaValue': '50'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '44s'}]}}
                # se e contem 'RESOURCE_EXHAUSTED'
                if "RESOURCE_EXHAUSTED" in str(e):
                    logging.error(f"Resource exhausted: {e}")
                    logging.warning("Waiting for 1minute before retrying...")
                    time.sleep(60)
                # checka se é keyboard e se sim, espera 30 segundos
                elif "KeyboardInterrupt" in str(e):
                    raise KeyboardInterrupt
                else:
                    logging.error(f"An error occurred: {e}")
                    logging.warning("Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                continue
            # logging.info(f"Response: {response}")
            
            try:
                model_answer = self.convert_answer_to_LLMAnswer(response.candidates[0].content.parts[0].text)
                return model_answer
            except ValueError as e:
                logging.error(f"Failed to convert answer to LLMAnswer: {e}")
                prompt = prompt + f"\nYou have already answered this question, but your answer was not in the correct format. Please answer again, but this time make sure to follow the instructions and return a valid JSON. Previous error: {e}"
        