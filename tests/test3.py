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

def cosine_similarity_score(text1, text2):
    logging.info("Calculando a similaridade do cosseno (TF-IDF).")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    score_value = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    logging.info(f"Similaridade do cosseno: {score_value:.4f}")
    return score_value

def jaccard_similarity_score(text1, text2):
    logging.info("Calculando a similaridade de Jaccard.")
    tokens1 = set(preprocess_text(text1))
    tokens2 = set(preprocess_text(text2))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    score_value = len(intersection) / len(union) if union else 0
    logging.info(f"Similaridade de Jaccard: {score_value:.4f}")
    return score_value

def ngram_similarity_score(text1, text2, n=2):
    logging.info(f"Calculando a similaridade de {n}-gramas.")
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    score_value = len(intersection) / len(union) if union else 0
    logging.info(f"Similaridade de {n}-gramas: {score_value:.4f}")
    return score_value

def fuzzy_similarity_score(text1, text2):
    logging.info("Calculando a similaridade fuzzy.")
    score_value = fuzz.ratio(text1, text2) / 100.0
    logging.info(f"Similaridade fuzzy: {score_value:.4f}")
    return score_value

def difflib_similarity_score(text1, text2):
    logging.info("Calculando a similaridade usando difflib.")
    score_value = difflib.SequenceMatcher(None, text1, text2).ratio()
    logging.info(f"Similaridade difflib: {score_value:.4f}")
    return score_value

def sbert_similarity_score(text1, text2):
    logging.info("Calculando a similaridade usando SBERT.")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score_value = util.pytorch_cos_sim(emb1, emb2).item()
    logging.info(f"Similaridade SBERT: {score_value:.4f}")
    return score_value

def bertscore_similarity_score(text1, text2):
    logging.info("Calculando a similaridade usando BERTScore.")
    P, R, F1 = bert_score(
        [text1], 
        [text2], 
        lang="pt",
        model_type="bert-base-multilingual-cased",
        verbose=False
    )
    score_value = F1.mean().item()
    logging.info(f"BERTScore Similarity: {score_value:.4f}")
    return score_value

def combined_similarity(text1, text2):
    logging.info("Calculando a similaridade combinada.")
    scores = {
        'Cosine Similarity (TF-IDF)': cosine_similarity_score(text1, text2),
        'Jaccard Similarity': jaccard_similarity_score(text1, text2),
        'Bigram Similarity': ngram_similarity_score(text1, text2, n=2),
        'Trigram Similarity': ngram_similarity_score(text1, text2, n=3),
        'Fuzzy Similarity': fuzzy_similarity_score(text1, text2),
        'Difflib Similarity': difflib_similarity_score(text1, text2),
        'SBERT Similarity': sbert_similarity_score(text1, text2),
        'BERTScore Similarity': bertscore_similarity_score(text1, text2)
    }
    # Combinação ponderada: 40% SBERT, 40% BERTScore, 20% média dos métodos lexicais
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
# text1 = "ok, tá liberado. segue em frente"
# text2 = "sim, você pode fazer isso. está correto"

# text1 = "vou mijar no banheiro"
# text2 = "urinarei no toalete"

text1 = "I'm going to pee in the bathroom."
text2 = "I will urinate in the toilet."

scores, combined_score = combined_similarity(text1, text2)
print("Exemplo 1:")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
print(f"\nCombined Similarity Score: {combined_score:.4f}\n")

# Exemplo 2
text1 = "vou mijar no banheiro"
text2 = "urinarei no toalete"
scores, combined_score = combined_similarity(text1, text2)
print("Exemplo 2:")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
print(f"\nCombined Similarity Score: {combined_score:.4f}")
