import json
import pickle
import copy
import logging
from datasets import load_dataset
from tqdm import tqdm
from pprint import pprint




class FreeFormAnswer:
    def __init__(self, free_form_answers, gold_answer=""):
        self.gold_answer = gold_answer
        # Garante que free_form_answers seja uma lista de strings
        if isinstance(free_form_answers, list):
            self.free_form_answers = [str(ans) for ans in free_form_answers if ans != ""]
        else:
            self.free_form_answers = [str(free_form_answers)] if free_form_answers != "" else []

    def to_dict(self):
        return {
            "gold_answer": self.gold_answer,
            "free_form_answers": self.free_form_answers
        }


class QuestionAnswer:
    def __init__(self, question, unanswerable, option_answers):
        self.question = question
        self.unanswerable = unanswerable
        if isinstance(option_answers, FreeFormAnswer):
            self.option_answers = option_answers
        else:
            self.option_answers = FreeFormAnswer(option_answers)

    def to_dict(self):
        return {
            "question": self.question,
            "unanswerable": self.unanswerable,
            "option_answers": self.option_answers.to_dict()
        }


class DatasetData:
    def __init__(self, title, abstract, full_text, qas, src_dataset):
        self.title = title
        self.abstract = abstract
        self.full_text = full_text
        self.qas = qas
        self.src_dataset = src_dataset

    def to_dict(self):
        return {
            "title": self.title,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "qas": [qa.to_dict() for qa in self.qas],
            "src_dataset": self.src_dataset
        }


class JoinedDataset:
    def __init__(self):
        self.dataset = []

    def to_dict(self):
        return {
            "dataset": [data.to_dict() for data in self.dataset]
        }

    def remove_full_text(self):
        for data in self.dataset:
            data.full_text = ""


def extract_free_form_answers(option):
    """
    Extrai respostas de uma opção do QASPER.
    Se o campo 'yes_no' estiver definido, retorna "yes" ou "no";
    caso contrário, concatena os campos 'free_form_answer' e 'extractive_spans'.
    """
    answers = []
    yes_no = option.get('yes_no')
    if yes_no is not None:
        answers.append("yes" if yes_no else "no")
    else:
        # Processa free_form_answer, que pode ser uma lista ou string
        ffa = option.get('free_form_answer')
        if isinstance(ffa, list):
            answers.extend(ffa)
        elif ffa:
            answers.append(ffa)
        # Processa extractive_spans, que pode ser lista ou string
        spans = option.get('extractive_spans')
        if isinstance(spans, list):
            sp_resp = ""
            for sp in spans:
                sp_resp += sp + ", "
            sp_resp = sp_resp[:-2]  # Remove a última vírgula
            answers.append(sp_resp)
        elif spans:
            answers.append(spans)
    return answers


def process_qasper(dataset, dataset_name):
    processed_data = []
    logging.info(f"Iniciando o processamento do dataset QASPER - {dataset_name}")
    for item in tqdm(dataset, desc=f"QASPER {dataset_name}"):
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        # Concatena todos os parágrafos de forma mais limpa
        paragraphs = item.get("full_text", {}).get('paragraphs', [])
        full_text = " ".join(" ".join(section) for section in paragraphs)
        qas = []
        questions = item.get("qas", {}).get("question", [])
        answers_list = item.get("qas", {}).get("answers", [])
        for question, answer in zip(questions, answers_list):
            unanswerable = answer.get('answer', [{}])[0].get("unanswerable", False)
            options = answer.get('answer', [])
            free_form_answers = []
            if not unanswerable:
                for option in options:
                    free_form_answers.extend(extract_free_form_answers(option))
            else:
                free_form_answers.append("N/A")
            qa = QuestionAnswer(question, unanswerable, free_form_answers)
            qas.append(qa)
        processed_data.append(DatasetData(title, abstract, full_text, qas, "qasper"))
    return processed_data


def process_quality(dataset, dataset_name):
    processed_data = []
    logging.info(f"Iniciando o processamento do dataset Quality - {dataset_name}")
    for item in tqdm(dataset, desc=f"Quality {dataset_name}"):
        title = item.get("title", "")
        full_text = item.get("article", "")
        question = item.get("question", "")
        options = item.get("options", [])
        # Usa as opções como respostas
        option_answers = FreeFormAnswer(options)
        qa = QuestionAnswer(question, False, option_answers)
        processed_data.append(DatasetData(title, "", full_text, [qa], "quality"))
    return processed_data


def main():
    logging.info("Carregando os datasets...")
    qasper = load_dataset("allenai/qasper")
    quality = load_dataset("tasksource/QuALITY")

    joined_dataset = JoinedDataset()

    # Processa datasets QASPER
    joined_dataset.dataset.extend(process_qasper(qasper["train"], "train"))
    joined_dataset.dataset.extend(process_qasper(qasper["validation"], "validation"))
    joined_dataset.dataset.extend(process_qasper(qasper["test"], "test"))

    # Processa dataset Quality
    joined_dataset.dataset.extend(process_quality(quality["train"], "train"))
    joined_dataset.dataset.extend(process_quality(quality["validation"], "validation"))

    # Salva o dataset completo
    with open("results/joined_dataset.json", "w", encoding="utf-8") as f:
        json.dump(joined_dataset.to_dict(), f, ensure_ascii=False, indent=2)
    logging.info("Arquivo joined_dataset.json salvo com sucesso.")

    # Cria uma cópia para salvar sem o full_text
    joined_dataset_no_full_text = copy.deepcopy(joined_dataset)
    joined_dataset_no_full_text.remove_full_text()
    with open("results/joined_dataset_no_full_text.json", "w", encoding="utf-8") as f:
        json.dump(joined_dataset_no_full_text.to_dict(), f, ensure_ascii=False, indent=2)
    logging.info("Arquivo joined_dataset_no_full_text.json salvo com sucesso.")

    # Salva em formato pickle
    with open("results/joined_dataset.pkl", "wb") as f:
        pickle.dump(joined_dataset, f)
    logging.info("Arquivo joined_dataset.pkl salvo com sucesso.")


if __name__ == '__main__':
    # Configuração do logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
