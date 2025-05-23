import nltk
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import difflib
import logging
from bert_score import score as bert_score
from langdetect import detect, DetectorFactory


import re
from num2words import num2words

sentense_transformer_model_sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def download_nltk_resources():
    """
    Verifica e baixa os recursos necessários do NLTK.
    """
    logging.info("Iniciando o download dos recursos do NLTK.")
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logging.info(f"Recurso '{resource}' já está disponível.")
        except LookupError:
            nltk.download(resource)
            logging.info(f"Recurso '{resource}' foi baixado com sucesso.")

# Baixa os recursos do NLTK, se necessário
download_nltk_resources()

def preprocess_text(text):
    """
    Pré-processa o texto: tokeniza, converte para minúsculas e remove stopwords e tokens não alfanuméricos.
    """
    logging.debug("Iniciando o pré-processamento do texto.")
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('portuguese')]
    logging.debug(f"Texto pré-processado: {filtered_tokens}")
    return filtered_tokens

def cosine_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade do cosseno usando TF-IDF, tratando casos de vocabulário vazio.
    """
    logging.debug("Calculando a similaridade do cosseno (TF-IDF).")
    # Configura o vectorizer para aceitar tokens com 1 ou mais caracteres
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        tfidf_matrix = vectorizer.fit_transform([reference_text, generated_text])
        # Se o vocabulário ficar vazio, força a exceção
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("similarityScoreModule.py - Empty vocabulary")
        score_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        logging.debug(f"Similaridade do cosseno: {score_value:.4f}")
    except ValueError as e:
        logging.error(f"similarityScoreModule.py - Erro no cálculo do TF-IDF: {e}")
        score_value = 0.0
    return score_value


def jaccard_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade de Jaccard entre dois textos.
    """
    logging.debug("Calculando a similaridade de Jaccard.")
    tokens1 = set(preprocess_text(reference_text))
    tokens2 = set(preprocess_text(generated_text))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    score_value = len(intersection) / len(union) if union else 0
    logging.debug(f"Similaridade de Jaccard: {score_value:.4f}")
    return score_value

def ngram_similarity_score(reference_text, generated_text, n=2):
    """
    Calcula a similaridade de n-gramas entre dois textos.
    """
    logging.debug(f"Calculando a similaridade de {n}-gramas.")
    tokens1 = preprocess_text(reference_text)
    tokens2 = preprocess_text(generated_text)
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    score_value = len(intersection) / len(union) if union else 0
    logging.debug(f"Similaridade de {n}-gramas: {score_value:.4f}")
    return score_value

def fuzzy_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade fuzzy entre dois textos.
    """
    logging.debug("Calculando a similaridade fuzzy.")
    score_value = fuzz.ratio(reference_text, generated_text) / 100.0
    logging.debug(f"Similaridade fuzzy: {score_value:.4f}")
    return score_value

def difflib_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade usando difflib entre dois textos.
    """
    logging.debug("Calculando a similaridade usando difflib.")
    score_value = difflib.SequenceMatcher(None, reference_text, generated_text).ratio()
    logging.debug(f"Similaridade difflib: {score_value:.4f}")
    return score_value

def sbert_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade usando SBERT.
    """
    logging.debug("Calculando a similaridade usando SBERT.")
    
    emb1 = sentense_transformer_model_sbert.encode(reference_text, convert_to_tensor=True, show_progress_bar = False)
    emb2 = sentense_transformer_model_sbert.encode(generated_text, convert_to_tensor=True, show_progress_bar = False)
    score_value = util.pytorch_cos_sim(emb1, emb2).item()
    logging.debug(f"Similaridade SBERT: {score_value:.4f}")
    return score_value

# Aumenta a consistência na detecção do idioma
DetectorFactory.seed = 0

def detect_language(text):
    """
    Detecta o idioma do texto.
    """
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Falha na detecção do idioma: {e}")
        return 'en'  # Fallback para inglês se a detecção falhar

def bertscore_similarity_score(reference_text, generated_text):
    """
    Calcula a similaridade usando BERTScore.
    """
    logging.debug("Calculando a similaridade usando BERTScore.")
    ref_lang = detect_language(reference_text)
    gen_lang = detect_language(generated_text)
    logging.debug(f"Idiomas detectados - Referência: {ref_lang}, Gerado: {gen_lang}")
    # Prioriza o idioma da referência
    main_lang = ref_lang
    P, R, F1 = bert_score(
        [reference_text],
        [generated_text],
        lang=main_lang,
        model_type="bert-base-multilingual-cased",
        verbose=False
    )
    score_value = F1.mean().item()
    logging.debug(f"BERTScore Similarity: {score_value:.4f}")
    return score_value



def numbers_to_words(text):
    # Substitui cada número inteiro encontrado por sua versão por extenso
    lang = detect_language(text)
    logging.debug(f"Convertendo números para palavras no idioma: {lang}")
    if not (lang == 'pt' or lang == 'en'):
        logging.warning(f"Idioma não suportado para conversão de números: {lang}. Usando 'en' como padrão.")
        lang = 'en'
    # Função auxiliar para substituir números por palavras
    def replace_number(match):
        number = int(match.group())
        return num2words(number, lang=lang)

    # Substitui todos os números inteiros no texto
    converted_text = re.sub(r'\b\d+\b', replace_number, text)
    return converted_text

def combined_similarity(reference_text, generated_text):
    """
    Calcula diversas métricas de similaridade entre dois textos e retorna um dicionário com os scores
    e uma pontuação combinada ponderada.
    """
    logging.info("Calculando a similaridade combinada.")
    reference_text = numbers_to_words(str(reference_text))
    generated_text = numbers_to_words(str(generated_text))
    scores = {
        'Cosine Similarity (TF-IDF)': cosine_similarity_score(reference_text, generated_text),
        'Jaccard Similarity': jaccard_similarity_score(reference_text, generated_text),
        'Bigram Similarity': ngram_similarity_score(reference_text, generated_text, n=2),
        'Trigram Similarity': ngram_similarity_score(reference_text, generated_text, n=3),
        'Fuzzy Similarity': fuzzy_similarity_score(reference_text, generated_text),
        'Difflib Similarity': difflib_similarity_score(reference_text, generated_text),
        'SBERT Similarity': sbert_similarity_score(reference_text, generated_text),
        'BERTScore Similarity': bertscore_similarity_score(reference_text, generated_text)
    }
    # Ponderação: 40% para SBERT, 40% para BERTScore, 20% para a média dos métodos lexicais (ANTIGO)
    # lexical_methods = [
    #     scores['Cosine Similarity (TF-IDF)'],
    #     scores['Jaccard Similarity'],
    #     scores['Bigram Similarity'],
    #     scores['Trigram Similarity'],
    #     scores['Fuzzy Similarity'],
    #     scores['Difflib Similarity']
    # ]
    # lexical_avg = np.mean(lexical_methods)
    
    # METRICA AGORA É SOMENTE A MÉDIA PONDERADA ENTRE SBERT E BERTSCORE (ANALISE DE SEMANTICA)
    combined_score_value = 0.5 * scores['SBERT Similarity'] + 0.5 * scores['BERTScore Similarity'] 
    logging.debug(f"Scores de similaridade: {scores}")
    logging.debug(f"Pontuação de similaridade combinada (ponderada): {combined_score_value:.4f}")
    return scores, combined_score_value

if __name__ == "__main__":
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Exemplos de uso quando o módulo é executado diretamente
    """
    4
    • four
    • usually a car has 4 wheels
    • if not broken, four
    • two in the front and two in the back
    \item Cars usually have 6 or more wheels depending on the model
    \item I think it depends on the season
    \item It has legs instead of wheels
    \item The engine is what makes it move
    \item Probably two, because bicycles have two
    \item There are two wheels in the front
    """
    # Exemplo 1
    reference_text = "Four Wheels"
    print(f"Referência: {reference_text}")
    generated_text = "4"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f"Exemplo 1: {generated_text}")
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}\n")

    # Exemplo 2
    generated_text = "Usually a car has 4 wheels"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 2: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    
    # Exemplo 3
    generated_text = "If not broken, four"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 3: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 4
    generated_text = "Two in the front and two in the back"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 4: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    
    # Exemplo 5
    generated_text = "Cars usually have 6 or more wheels depending on the model"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 5: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 6
    generated_text = "I think it depends on the season"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 6: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 7
    generated_text = "It has legs instead of wheels"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 7: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 8
    generated_text = "The engine is what makes it move"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 8: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 9
    generated_text = "Probably two, because bicycles have two"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 9: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    # Exemplo 10
    generated_text = "There are two wheels in the front"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print(f'Exemplo 10: {generated_text}')
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
    