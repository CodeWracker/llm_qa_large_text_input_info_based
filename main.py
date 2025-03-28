from datasetModule import JoinedDataset, DatasetData, QuestionAnswer, FreeFormAnswer
from similarityScoreModule import sbert_similarity_score, combined_similarity


import pickle
import logging
import json

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name

    # virtual method to be implemented by subclasses
    def generate_answer(self, question):
        raise NotImplementedError("Subclasses should implement this method")
        

class Gemini1_5Flash(LLMModel):
    def __init__(self):
        super().__init__("Gemini 1.5 Flash")

    def generate_answer(self, question):
        # Implement the logic to generate an answer using the Gemini 1.5 Flash model
        # For example, you can use the model's API to get the answer for the question.
        # model_answer = "Generated answer from Gemini 1.5 Flash model"
        model_answer = "The seed lexicon contains a set of predicates with positive and negative meanings."
        return model_answer

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
            self.scores.append({"question": gt_answer, "scores": scores})
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
        

if __name__ == "__main__":
    # This file will create the data of the LLM generated data and score it against the ground truth dataset
    # load the dataset.plk file
    joined_dataset = pickle.load(open("joined_dataset.pkl", "rb"))
    
    gemini1_5flash_model = Gemini1_5Flash()
    llm_generated_dataset = []
    
    for data in joined_dataset.dataset:
        # Generate answers for each question in the dataset using the LLM model
        for qa in data.qas:
            llm_answer = gemini1_5flash_model.generate_answer(qa.question)
            gemini1_5flash_answer_obj = LLMAnswer(gemini1_5flash_model.model_name, llm_answer, qa.option_answers.free_form_answers)
            gemini1_5flash_answer_obj.calculate_similarity_score()
            comparison_result = ComparisonResult(qa)
            comparison_result.models_answers.append(gemini1_5flash_answer_obj)
            llm_generated_dataset.append(comparison_result)
            break
        break
    
    # Save the generated dataset with similarity scores
    with open("llm_generated_dataset.pkl", "wb") as f:
        pickle.dump(llm_generated_dataset, f)
    logger.info("LLM generated dataset saved successfully.")
    
    # Save as JSON file
    with open("llm_generated_dataset.json", "w", encoding="utf-8") as f:
        json_data = []
        for comparison_result in llm_generated_dataset:
            json_data.append(comparison_result.to_dict())
        json.dump(json_data, f, ensure_ascii=False, indent=2)