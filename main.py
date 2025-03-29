from datasetModule import JoinedDataset, DatasetData, QuestionAnswer, FreeFormAnswer
from similarityScoreModule import sbert_similarity_score, combined_similarity

from models.LLModel import LLMModel
from models.SpecificModels import (
    Gemini1_5Flash,
    Gemini1_5Flash8B,
    Gemini1_5Pro,
    Gemini2_0Flash,
    Gemini2_0FlashLite,
    Gemini2_0FlashThinkingExperimental,
    Gemini2_0FlashExperimental
)

import pickle
import logging
import json
from google import genai
import os


# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GeminiClient = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

class LLMAnswer:
    def __init__(self, model_name, answer, option_answers_gt):
        self.model_name = model_name
        self.answer = answer
        self.option_answers_gt = option_answers_gt
        self.similarity_score = None  # Similaridade geral (maior valor dentre as comparações)
        self.scores = []  # Lista de dicionários: { "gt_compared_answer": <gt>, "scores": <scores_dict> }

    def calculate_similarity_score(self):
        for gt_answer in self.option_answers_gt:
            scores, combined_score_value = combined_similarity(self.answer, gt_answer)
            logging.info(f"Similarity score between generated answer and ground truth answer: {combined_score_value:.4f}")
            self.scores.append({
                "gt_compared_answer": gt_answer,
                "scores": scores
            })
            if self.similarity_score is None or combined_score_value > self.similarity_score:
                self.similarity_score = combined_score_value

    def to_dict(self):
        # Transforma a lista de scores em um dicionário onde a chave é a resposta GT
        similarities = { item["gt_compared_answer"]: item["scores"] for item in self.scores }
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "overall_similarity": self.similarity_score,
            "similarities": similarities
        }

class ComparisonResult:
    def __init__(self, ground_truth_qas, text_title):
        self.question = ground_truth_qas.question
        self.text_title = text_title
        self.unanswerable = ground_truth_qas.unanswerable
        self.ground_truth = ground_truth_qas.option_answers.free_form_answers
        self.model_results = []  # Lista de instâncias de LLMAnswer

    def to_dict(self):
        model_results_list = [model_answer.to_dict() for model_answer in self.model_results]
        return {
            "question": self.question,
            "text_title": self.text_title,
            "unanswerable": self.unanswerable,
            "ground_truth": self.ground_truth,
            "model_results": model_results_list
        }

def process_models_for_comparison(comparison_results, question, base_text, client):
    # Lista dos modelos a serem processados
    modelos = [
        Gemini1_5Flash,
        Gemini1_5Flash8B,
        Gemini1_5Pro,
        Gemini2_0Flash,
        Gemini2_0FlashLite,
        Gemini2_0FlashThinkingExperimental,
        Gemini2_0FlashExperimental
    ]
    
    for modelo_class in modelos:
        # Instancia o modelo para obter o nome
        modelo_inst = modelo_class()
        model_name = modelo_inst.model_name
        # Verifica se esse modelo já foi processado para esse par
        if not any(mr.model_name == model_name for mr in comparison_results.model_results):
            logging.info(f"Processando modelo: {model_name} para a questão: {question}")
            answer_obj = modelo_inst.generate_answer(question, base_text, client)
            llm_answer_obj = LLMAnswer(model_name, answer_obj.answer, comparison_results.ground_truth)
            llm_answer_obj.calculate_similarity_score()
            comparison_results.model_results.append(llm_answer_obj)
        else:
            logging.info(f"Modelo {model_name} já processado para essa questão, pulando.")
    return comparison_results

if __name__ == "__main__":
    try:
        # Carrega o dataset original
        joined_dataset = pickle.load(open("results/joined_dataset.pkl", "rb"))
        
        # Tenta carregar os resultados gerados anteriormente
        results_path = "results/llm_generated_dataset.pkl"
        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                loaded_results = pickle.load(f)
            # Se os resultados carregados forem uma lista, converte para dicionário com chave "title+question"
            if isinstance(loaded_results, list):
                results_dict = {}
                for item in loaded_results:
                    key = f"{item.text_title}+{item.question}"
                    results_dict[key] = item
            else:
                results_dict = loaded_results
            logging.info("Resultados carregados com sucesso.")
        else:
            results_dict = {}
            logging.info("Nenhum resultado anterior encontrado. Processando tudo do zero.")
        
        # Processa o dataset e atualiza apenas os pares que faltam
        qtd_textos = len(joined_dataset.dataset)
        for data in joined_dataset.dataset:
            logging.info(f"Processando texto: {data.title} ({qtd_textos} textos restantes)")
            for qa in data.qas:
                key = f"{data.title}+{qa.question}"
                if key in results_dict:
                    comparison_result = results_dict[key]
                    logging.info(f"Par já processado: {key}. Verificando modelos faltantes.")
                else:
                    comparison_result = ComparisonResult(qa, data.title)
                
                # Processa apenas os modelos que ainda não foram computados para esse par
                comparison_result = process_models_for_comparison(comparison_result, qa.question, data.full_text, GeminiClient)
                
                # Atualiza ou adiciona o resultado no dicionário
                results_dict[key] = comparison_result
            
        
        # Converte os resultados para lista para salvar
        llm_generated_dataset = list(results_dict.values())
        
        # Salva o dataset gerado com as similaridades em formato pickle
        with open(results_path, "wb") as f:
            pickle.dump(results_dict, f)
        logging.info("LLM generated dataset salvo com sucesso (pickle).")
        
        # Salva como arquivo JSON com a nova estrutura
        with open("results/llm_generated_dataset.json", "w", encoding="utf-8") as f:
            json_data = [cr.to_dict() for cr in llm_generated_dataset]
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logging.info("LLM generated dataset salvo com sucesso (JSON).")
    except Exception as e:
        logging.error("Erro durante o processamento: %s", e)
        
    # Fechamento do GeminiClient
    try:
        GeminiClient._api_client._httpx_client.close()
    except Exception as e:
        logging.warning("Erro ao fechar o httpx client: %s", e)

    GeminiClient.__del__ = lambda self: None
    GeminiClient = None
