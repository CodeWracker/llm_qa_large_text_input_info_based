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
            raise ValueError("Empty vocabulary")
        score_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        logging.debug(f"Similaridade do cosseno: {score_value:.4f}")
    except ValueError as e:
        logging.error(f"Erro no cálculo do TF-IDF: {e}")
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

def combined_similarity(reference_text, generated_text):
    """
    Calcula diversas métricas de similaridade entre dois textos e retorna um dicionário com os scores
    e uma pontuação combinada ponderada.
    """
    logging.info("Calculando a similaridade combinada.")
    reference_text = str(reference_text)
    generated_text = str(generated_text)
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
    logging.info(f"Pontuação de similaridade combinada (ponderada): {combined_score_value:.4f}")
    return scores, combined_score_value

if __name__ == "__main__":
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Exemplos de uso quando o módulo é executado diretamente

    # Exemplo 1
    reference_text = "ok, tá liberado. segue em frente"
    generated_text = "sim, você pode fazer isso. está correto"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print("Exemplo 1:")
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}\n")

    # Exemplo 2
    reference_text = "vou mijar no banheiro"
    generated_text = "urinarei no toalete"
    scores, combined_score_value = combined_similarity(reference_text, generated_text)
    print("Exemplo 2:")
    for method, score in scores.items():
        print(f"{method}: {score:.4f}")
    print(f"\nCombined Similarity Score: {combined_score_value:.4f}")
