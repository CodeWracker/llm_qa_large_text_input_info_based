from models.LLModel import LLMModel,LLLAnswer

from google import genai
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Gemini1_5Flash(LLMModel):
    def __init__(self):
        super().__init__("Gemini 1.5 Flash")
        self.client = genai.ModelServiceClient()
        
        

    def generate_answer(self, question, base_text):
        # Implement the logic to generate an answer using the Gemini 1.5 Flash model
        # For example, you can use the model's API to get the answer for the question.
        # model_answer = "Generated answer from Gemini 1.5 Flash model"
        model_answer = "The seed lexicon contains a set of predicates with positive and negative meanings."

        
        prompt = self.generate_prompt(question, base_text)
        logger.info(f"Prompt: {prompt}")
        response = self.client.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': LLLAnswer
            }
        )
        logger.info(f"Response: {response}")
        
        
        
        return model_answer