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
    def query_model(self, prompt, model_name_query):
        """
        Send the prompt to a model until a valid JSON answer is returned.
        """
        raise NotImplementedError("Subclasses should implement this method")
        
    def generate_answer(self, question, base_text, model_name_query):
        prompt = self.generate_prompt(question, base_text)
        response = self.query_model(prompt, model_name_query)
        return response
        

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
    
    # virtual method to be implemented by subclasses
    def query_model(self, prompt, model_name_query):
        """
        Send the prompt to a model until a valid JSON answer is returned.
        """
        raise NotImplementedError("Subclasses should implement this method")
        
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
    def generate_answer(self, question, base_text, model_name_query):
        prompt = self.generate_prompt(question, base_text)

        return self.query_model(prompt,model_name_query)
