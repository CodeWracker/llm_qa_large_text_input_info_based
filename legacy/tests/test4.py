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

# Para BERTScore, certifique-se de instalar: pip install bert-score
from bert_score import score as bert_score
from langdetect import detect, DetectorFactory
# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Verifica e baixa os recursos necessários do NLTK
def download_nltk_resources():
    logging.info("Iniciando o download dos recursos do NLTK.")
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logging.info(f"Recurso '{resource}' já está disponível.")
        except LookupError:
            nltk.download(resource)
            logging.info(f"Recurso '{resource}' foi baixado com sucesso.")

download_nltk_resources()

def preprocess_text(text):
    logging.info("Iniciando a pré-processamento do texto.")
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('portuguese')]
    logging.info(f"Texto pré-processado: {filtered_tokens}")
    return filtered_tokens

def cosine_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade do cosseno (TF-IDF).")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference_text, generated_text])
    score_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    logging.info(f"Similaridade do cosseno: {score_value:.4f}")
    return score_value

def jaccard_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade de Jaccard.")
    tokens1 = set(preprocess_text(reference_text))
    tokens2 = set(preprocess_text(generated_text))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    score_value = len(intersection) / len(union) if union else 0
    logging.info(f"Similaridade de Jaccard: {score_value:.4f}")
    return score_value

def ngram_similarity_score(reference_text, generated_text, n=2):
    logging.info(f"Calculando a similaridade de {n}-gramas.")
    tokens1 = preprocess_text(reference_text)
    tokens2 = preprocess_text(generated_text)
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    score_value = len(intersection) / len(union) if union else 0
    logging.info(f"Similaridade de {n}-gramas: {score_value:.4f}")
    return score_value

def fuzzy_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade fuzzy.")
    score_value = fuzz.ratio(reference_text, generated_text) / 100.0
    logging.info(f"Similaridade fuzzy: {score_value:.4f}")
    return score_value

def difflib_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade usando difflib.")
    score_value = difflib.SequenceMatcher(None, reference_text, generated_text).ratio()
    logging.info(f"Similaridade difflib: {score_value:.4f}")
    return score_value

def sbert_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade usando SBERT.")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb1 = model.encode(reference_text, convert_to_tensor=True)
    emb2 = model.encode(generated_text, convert_to_tensor=True)
    score_value = util.pytorch_cos_sim(emb1, emb2).item()
    logging.info(f"Similaridade SBERT: {score_value:.4f}")
    return score_value

# Aumenta a consistência na detecção (opcional)
DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Fallback para inglês se a detecção falhar

def bertscore_similarity_score(reference_text, generated_text):
    logging.info("Calculando a similaridade usando BERTScore.")
    
    # Detecta o idioma de cada texto
    ref_lang = detect_language(reference_text)
    gen_lang = detect_language(generated_text)
    
    logging.info(f"Idiomas detectados - Referência: {ref_lang}, Gerado: {gen_lang}")
    
    # Escolhe o idioma principal (prioriza o do texto de referência)
    main_lang = ref_lang if ref_lang == gen_lang else ref_lang
    
    P, R, F1 = bert_score(
        [reference_text], 
        [generated_text], 
        lang=main_lang,
        model_type="bert-base-multilingual-cased",
        verbose=False
    )
    
    score_value = F1.mean().item()
    logging.info(f"BERTScore Similarity: {score_value:.4f}")
    return score_value

def combined_similarity(reference_text, generated_text):
    logging.info("Calculando a similaridade combinada.")
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
    # Se preferir, você pode dar mais peso aos métodos semânticos (SBERT e BERTScore)
    # Exemplo: ponderando 40% para SBERT, 40% para BERTScore e 20% para a média dos demais
    semantic_avg = (scores['SBERT Similarity'] + scores['BERTScore Similarity']) / 2
    lexical_methods = [
        scores['Cosine Similarity (TF-IDF)'],
        scores['Jaccard Similarity'],
        scores['Bigram Similarity'],
        scores['Trigram Similarity'],
        scores['Fuzzy Similarity'],
        scores['Difflib Similarity']
    ]
    lexical_avg = np.mean(lexical_methods)
    combined_score = 0.4 * scores['SBERT Similarity'] + 0.4 * scores['BERTScore Similarity'] + 0.2 * lexical_avg
    logging.info(f"Scores de similaridade: {scores}")
    logging.info(f"Pontuação de similaridade combinada (ponderada): {combined_score:.4f}")
    return scores, combined_score

# Exemplos de uso
# Exemplo 1
reference_text = "ok, tá liberado. segue em frente"
generated_text = "sim, você pode fazer isso. está correto"
scores, combined_score = combined_similarity(reference_text, generated_text)
print("Exemplo 1:")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
print(f"\nCombined Similarity Score: {combined_score:.4f}\n")

# Exemplo 2
reference_text = "vou mijar no banheiro"
generated_text = "urinarei no toalete"
scores, combined_score = combined_similarity(reference_text, generated_text)
print("Exemplo 2:")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
print(f"\nCombined Similarity Score: {combined_score:.4f}")
