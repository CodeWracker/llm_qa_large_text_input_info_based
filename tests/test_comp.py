from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modelo de embeddings para português
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Usando um modelo NLI alternativo
nli_model = pipeline(
    "text-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    tokenizer="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    return_all_scores=True
)

def evaluate_similarity(text1, text2):
    embedding1 = embedding_model.encode([text1])
    embedding2 = embedding_model.encode([text2])
    sim = cosine_similarity(embedding1, embedding2)
    return sim[0][0] * 100

def get_entailment_score(text1, text2):
    result = nli_model(f"{text1} </s> {text2}")
    scores = {entry['label']: entry['score'] for entry in result[0]}
    return scores.get("ENTAILMENT", 0) * 100

def evaluate_semantic_meaning(text1, text2):
    score1 = get_entailment_score(text1, text2)
    score2 = get_entailment_score(text2, text1)
    return (score1 + score2) / 2

def main():
    text1 = input("Digite o primeiro texto: ")
    text2 = input("Digite o segundo texto: ")
    
    similarity = evaluate_similarity(text1, text2)
    semantic_score = evaluate_semantic_meaning(text1, text2)
    
    print(f"\nSimilaridade do texto (cosine similarity): {similarity:.2f}%")
    print(f"Avaliação de significado (média de entailment): {semantic_score:.2f}%")
    
if __name__ == "__main__":
    main()
