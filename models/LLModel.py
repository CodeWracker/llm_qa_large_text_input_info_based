from pydantic import BaseModel

class LLLAnswer(BaseModel):
    question: str
    unanswerable: bool
    answer: str
    

class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name

    # virtual method to be implemented by subclasses
    def generate_answer(self, question, base_text):
        raise NotImplementedError("Subclasses should implement this method")
        

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