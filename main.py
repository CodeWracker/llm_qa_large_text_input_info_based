from datasetModule import JoinedDataset, DatasetData, QuestionAnswer, FreeFormAnswer
from similarityScoreModule import sbert_similarity_score, combined_similarity

from models.LLModel import LLMModel
from models.SpecificModels import Gemini1_5Flash, Gemini1_5Flash8B, Gemini1_5Pro, Gemini2_0Flash, Gemini2_0FlashLite, Gemini2_0FlashThinkingExperimental, Gemini2_0FlashExperimental


import pickle
import logging
import json
from google import genai
import os

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



GeminiClient = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


class LLMAnswer:
    def __init__(self, model_name, answer, option_answers_gt):
        self.model_name = model_name
        self.answer = answer
        self.option_answers_gt = option_answers_gt
        self.similarity_score = None
        self.scores = []
    def calculate_similarity_score(self):
        for gt_answer in self.option_answers_gt:
            
            scores, combined_score_value = combined_similarity(self.answer, gt_answer)
            logger.info(f"Similarity score between generated answer and ground truth answer: {combined_score_value:.4f}")
            self.scores.append({"gt_compared_answer": gt_answer, "scores": scores})
            if self.similarity_score is None or combined_score_value > self.similarity_score:
                self.similarity_score = combined_score_value
        pass
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "similarity_score": self.similarity_score,
            "scores": self.scores           
        }

class ComparisonResult:
    def __init__(self, groud_truth_qas):
        self.question = groud_truth_qas.question
        self.unanswerable = groud_truth_qas.unanswerable
        self.option_answers_gt = groud_truth_qas.option_answers.free_form_answers
        self.models_answers = []
    def to_dict(self):
        dict_model_answers = [model_answer.to_dict() for model_answer in self.models_answers]
        return {
            "question": self.question,
            "unanswerable": self.unanswerable,
            "option_answers_gt": self.option_answers_gt,
            "dict_model_answers": dict_model_answers
        }
 
def generate_models_answer(qa, base_text, client):
    question = qa.question
    comparison_results = ComparisonResult(qa)
    
    
    gemini1_5flash_model = Gemini1_5Flash()
    gemini1_5flash_answer = gemini1_5flash_model.generate_answer(question, base_text, client).answer
    gemini1_5flash_answer_obj = LLMAnswer(gemini1_5flash_model.model_name, gemini1_5flash_answer, qa.option_answers.free_form_answers)
    gemini1_5flash_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini1_5flash_answer_obj)
    
    gemini1_5flash8b_model = Gemini1_5Flash8B()
    gemini1_5flash8b_answer = gemini1_5flash8b_model.generate_answer(question, base_text, client).answer
    gemini1_5flash8b_answer_obj = LLMAnswer(gemini1_5flash8b_model.model_name, gemini1_5flash8b_answer, qa.option_answers.free_form_answers)
    gemini1_5flash8b_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini1_5flash8b_answer_obj)
    
    
    gemini1_5pro_model = Gemini1_5Pro()
    gemini1_5pro_answer = gemini1_5pro_model.generate_answer(question, base_text, client).answer
    gemini1_5pro_answer_obj = LLMAnswer(gemini1_5pro_model.model_name, gemini1_5pro_answer, qa.option_answers.free_form_answers)
    gemini1_5pro_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini1_5pro_answer_obj)
    
    gemini2_0flash_model = Gemini2_0Flash()
    gemini2_0flash_answer = gemini2_0flash_model.generate_answer(question, base_text, client).answer
    gemini2_0flash_answer_obj = LLMAnswer(gemini2_0flash_model.model_name, gemini2_0flash_answer, qa.option_answers.free_form_answers)
    gemini2_0flash_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini2_0flash_answer_obj)
    
    gemini2_0flashlite_model = Gemini2_0FlashLite()
    gemini2_0flashlite_answer = gemini2_0flashlite_model.generate_answer(question, base_text, client).answer
    gemini2_0flashlite_answer_obj = LLMAnswer(gemini2_0flashlite_model.model_name, gemini2_0flashlite_answer, qa.option_answers.free_form_answers)
    gemini2_0flashlite_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini2_0flashlite_answer_obj)
    
    gemini2_0flashthinking_model = Gemini2_0FlashThinkingExperimental()
    gemini2_0flashthinking_answer = gemini2_0flashthinking_model.generate_answer(question, base_text, client).answer
    gemini2_0flashthinking_answer_obj = LLMAnswer(gemini2_0flashthinking_model.model_name, gemini2_0flashthinking_answer, qa.option_answers.free_form_answers)
    gemini2_0flashthinking_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini2_0flashthinking_answer_obj)
    
    gemini2_0flashexperimental_model = Gemini2_0FlashExperimental()
    gemini2_0flashexperimental_answer = gemini2_0flashexperimental_model.generate_answer(question, base_text, client).answer
    gemini2_0flashexperimental_answer_obj = LLMAnswer(gemini2_0flashexperimental_model.model_name, gemini2_0flashexperimental_answer, qa.option_answers.free_form_answers)
    gemini2_0flashexperimental_answer_obj.calculate_similarity_score()
    comparison_results.models_answers.append(gemini2_0flashexperimental_answer_obj)

    return comparison_results
    
    

if __name__ == "__main__":
    # This file will create the data of the LLM generated data and score it against the ground truth dataset
    # load the dataset.plk file
    joined_dataset = pickle.load(open("results/joined_dataset.pkl", "rb"))
    
    llm_generated_dataset = []
    
    for data in joined_dataset.dataset:
        # Generate answers for each question in the dataset using the LLM model
        for qa in data.qas:
            comparison_results = generate_models_answer(qa, data.full_text, GeminiClient)
            llm_generated_dataset.append(comparison_results)
            break
        break
    
    # Save the generated dataset with similarity scores
    with open("results/llm_generated_dataset.pkl", "wb") as f:
        pickle.dump(llm_generated_dataset, f)
    logger.info("LLM generated dataset saved successfully.")
    
    # Save as JSON file
    with open("results/llm_generated_dataset.json", "w", encoding="utf-8") as f:
        json_data = []
        for comparison_result in llm_generated_dataset:
            json_data.append(comparison_result.to_dict())
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
    # logging.info("Attempting to delete GeminiClient...")
    # https://github.com/googleapis/python-genai/issues/588
    # GeminiClient = None
    # GeminiClient._api_client._httpx_client.close()
    # GeminiClient._api_client._httpx_client._state = ClientState.CLOSED
    
    # Explicitly close the underlying httpx client
    try:
        GeminiClient._api_client._httpx_client.close()
    except Exception as e:
        logger.warning("Error closing httpx client: %s", e)

    # Optionally override the __del__ method to a no-op to bypass cleanup code
    GeminiClient.__del__ = lambda self: None

    # Now set GeminiClient to None, so the original __del__ isn’t called
    GeminiClient = None
    
    # logger.info("GeminiClient closed successfully.")
   