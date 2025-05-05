from datasetModule import JoinedDataset, DatasetData, QuestionAnswer, FreeFormAnswer
from similarityScoreModule import sbert_similarity_score, combined_similarity
from ai_jury import ask_ai_jury

from models.GoogleModels import (
    Gemini1_5Flash,
    Gemini1_5Flash8B,
    Gemini1_5Pro,
    Gemini2_0Flash,
    Gemini2_0FlashLite,
    Gemini2_0FlashThinkingExperimental,
    Gemini2_0FlashExperimental
)

from models.OllamaModels import (
    DeepSeekR1_1_5b,
    Llama32_1b,
    Gemma3_1b
)


import pickle
import logging
import json
import os
import time


# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configuração do logging
# Configuração para log em arquivo
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='main.py.log',
    filemode='a'
)

# Adiciona um handler para o console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)




class AiJuryModelOpinion:
    def __init__(self, model_name, opinion):
        self.is_correct = opinion["is_correct"]
        self.justification = opinion["justification"]

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "opinion": self.opinion
        }

class AiJury:
    def __init__(self, question, reference_answer, eval_answer):
        self.question = question
        self.reference_answer = reference_answer
        self.eval_answer = eval_answer

    def ask_ai_jury(self):
        veredict = ask_ai_jury(self.question, self.reference_answer, self.eval_answer)
        self.final_verdict = veredict["final_verdict"]
        self.models_opinion = []
        for model in veredict["models_opinion"].keys():
            model_name = model
            opinion = veredict["models_opinion"][model]
            self.models_opinion.append(AiJuryModelOpinion(model_name, opinion))
            
    def to_dict(self):
        return {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "eval_answer": self.eval_answer,
            "final_verdict": self.final_verdict,
            "models_opinion": [opinion.to_dict() for opinion in self.models_opinion]
        }
class LLMAnswer:
    def __init__(self, model_name, answer, option_answers_gt, question=None):
        self.question = question
        self.model_name = model_name
        self.answer = answer
        self.option_answers_gt = option_answers_gt
        self.similarity_score = None  # Similaridade geral (maior valor dentre as comparações)
        self.scores = []  # Lista de dicionários: { "gt_compared_answer": <gt>, "scores": <scores_dict> }
        logging.info(f"Instância de LLMAnswer criada para o modelo: {model_name} - Resposta: {answer}")

    def calculate_similarity_score(self):
        for gt_answer in self.option_answers_gt:
            # Verifica se a resposta do modelo ou o ground truth são "N/A"
            answer_is_na = self.answer.strip().upper() == "N/A"
            gt_is_na = gt_answer.strip().upper() == "N/A"
            
            if answer_is_na or gt_is_na:
                # Se ao menos uma for "N/A", não calculamos as métricas detalhadas (scores será um objeto vazio)
                scores = {}
                # Se a resposta do modelo for "N/A" e o ground truth não for, pontuação 0
                if answer_is_na and not gt_is_na:
                    combined_score_value = 0.0
                # Se ambas forem "N/A", pontuação 1
                elif answer_is_na and gt_is_na:
                    combined_score_value = 1.0
                # Caso não especificado (por exemplo, ground truth for "N/A" e a resposta não), define como 0
                else:
                    combined_score_value = 0.0
            else:
                scores, combined_score_value = combined_similarity(self.answer, gt_answer)
            ai_jury_veredict_result = ask_ai_jury(self.answer, gt_answer, self.question)
                
            
            logging.info(f"Similarity score between generated answer and ground truth answer: {combined_score_value:.4f}")
            
            
            
            self.scores.append({
                "gt_compared_answer": gt_answer,
                "scores": scores,
                "ai_jury_veredict_result": ai_jury_veredict_result
            })
            if self.similarity_score is None or combined_score_value > self.similarity_score:
                self.similarity_score = combined_score_value


    def to_dict(self):
        # Transforma a lista de scores em um dicionário onde a chave é a resposta GT
        similarities = { item["gt_compared_answer"]: item["scores"] for item in self.scores }
        # check if ai jury results are empty - necessary because this was added after the first implementation
        ai_jury_results ={}
        for score in self.scores:
            #check if ai_jury_veredict_result key exists in score
            if not "ai_jury_veredict_result" in score:
                score["ai_jury_veredict_result"] = {}

            ai_jury_results[score["gt_compared_answer"]] = score["ai_jury_veredict_result"]
            
            
        
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "overall_similarity": self.similarity_score,
            "similarities": similarities,
            "ai_jury_results": ai_jury_results
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

def process_models_for_comparison(comparison_results, question, base_text):
    # Lista dos modelos a serem processados
    modelos = [
        # the context is too big for the models from Ollama, it can't process and ends up hallucinating
        # DeepSeekR1_1_5b,
        # Llama32_1b,
        # Gemma3_1b,
        Gemini1_5Flash,
        Gemini1_5Flash8B,
        Gemini1_5Pro,
        Gemini2_0Flash,
        Gemini2_0FlashLite,
        Gemini2_0FlashThinkingExperimental,
        Gemini2_0FlashExperimental,
    ]
    
    for modelo_class in modelos:
        # Instancia o modelo para obter o nome
        modelo_inst = modelo_class()
        model_name = modelo_inst.model_name
        # Verifica se esse modelo já foi processado para esse par
        if not any(mr.model_name == model_name for mr in comparison_results.model_results):
            logging.info(f"Processando modelo: {model_name} para a questão: {question}")
            answer_obj = modelo_inst.generate_answer(question, base_text)
            llm_answer_obj = LLMAnswer(model_name, answer_obj.answer, comparison_results.ground_truth, question)
            llm_answer_obj.calculate_similarity_score()
            comparison_results.model_results.append(llm_answer_obj)
        else:
            logging.info(f"Modelo {model_name} já processado para essa questão, pulando a geração da resposta.")
            logging.info(f"Recalculando a similaridade para o modelo {model_name}.")
            # Recalcula a similaridade para o modelo já processado
            for mr in comparison_results.model_results:
                mr.question = question # precisa atualizar a pergunta
                if mr.model_name == model_name:
                    mr.calculate_similarity_score()
                    break
    return comparison_results

def save_checkpoint(results_dict, results_path):
    # Salva o checkpoint em formato pickle
    with open(results_path, "wb") as f:
        pickle.dump(results_dict, f)
    # Salva o checkpoint em formato JSON
    with open("results/llm_generated_dataset.json", "w", encoding="utf-8") as f:
        json_data = [cr.to_dict() for cr in list(results_dict.values())]
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logging.info("Checkpoint salvo com sucesso.")

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
            qtd_carregados = len(results_dict)
            logging.info(f"Resultados carregados: {qtd_carregados} pares já processados.")
        else:
            results_dict = {}
            logging.info("Nenhum resultado anterior encontrado. Processando tudo do zero.")
        
        # Inicia a contagem de tempo e define o total de perguntas a serem processadas
        start_time = time.time()
        total_questions = sum(len(data.qas) for data in joined_dataset.dataset)
        questions_processed = 0

        # Lista para armazenar os tempos dos últimos 5 processamentos
        last_times = []

        qtd_textos = len(joined_dataset.dataset)
        for data in joined_dataset.dataset:
            qtd_textos -= 1
            logging.info(f"Processando texto: {data.title} ({qtd_textos} textos restantes)")
            for qa in data.qas:
                key = f"{data.title}+{qa.question}"
                if key in results_dict:
                    comparison_result = results_dict[key]
                    logging.info(f"Par já processado: {key}. Verificando modelos faltantes.")
                else:
                    comparison_result = ComparisonResult(qa, data.title)
                
                # Inicia a medição do tempo para a pergunta atual
                question_start = time.time()

                # Processa apenas os modelos que ainda não foram computados para esse par
                while True:
                    try:
                        comparison_result = process_models_for_comparison(
                            comparison_result, qa.question, data.full_text
                        )
                        break
                    except KeyboardInterrupt:
                        logging.info("main.py - Processamento interrompido pelo usuário.")
                        raise KeyboardInterrupt
                    except Exception as e:
                        logging.error(f"main.py - Erro ao processar o modelo: {e}")
                        logging.info("Tentando novamente após erro.")
                        time.sleep(2)
                        continue
                
                # Atualiza ou adiciona o resultado no dicionário
                results_dict[key] = comparison_result

                # Salva checkpoint após cada pergunta processada
                save_checkpoint(results_dict, results_path)
                
                # Atualiza o contador de perguntas processadas
                questions_processed += 1
                
                # Calcula o tempo gasto para esta pergunta e atualiza a lista dos últimos tempos
                question_elapsed = time.time() - question_start
                last_times.append(question_elapsed)
                if len(last_times) > 5:
                    last_times.pop(0)
                
                # Calcula o tempo médio com base nos últimos 5 processamentos (ou menos, se ainda não houver 5)
                average_time = sum(last_times) / len(last_times)
                remaining_questions = total_questions - questions_processed
                estimated_remaining_time = average_time * remaining_questions
                
                elapsed_time_total = time.time() - start_time
                
                # Log de tempo decorrido e estimativa de tempo restante
                logging.info(
                    f"Tempo total gasto até agora: {elapsed_time_total:.2f} segundos. "
                    f"Estimativa de tempo restante: {estimated_remaining_time:.2f} segundos."
                )
                
                # Log do progresso de perguntas processadas
                percent_done = (questions_processed / total_questions) * 100
                logging.info(
                    f"{questions_processed} perguntas respondidas por todos os modelos de um total de {total_questions}. "
                    f"Processamento em {percent_done:.2f}%."
                )
            logging.info(f"Texto {data.title} processado com sucesso com {len(data.qas)} perguntas respondidas.")
        logging.info("Processamento completo do dataset.")
                
    except KeyboardInterrupt as e:
        logging.error(f"main.py - Processamento interrompido pelo usuário.")
        
    
