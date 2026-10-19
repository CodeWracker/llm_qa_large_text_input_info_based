from sentence_transformers import SentenceTransformer, util
import tensorflow_text
import tensorflow_hub as hub
import numpy as np



model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sent1 = "ok, ta liberado. segue em frente"
sent2 = "sim, voce pode fazer isso. esta correto"

emb1 = model.encode(sent1, convert_to_tensor=True)
emb2 = model.encode(sent2, convert_to_tensor=True)

similarity = util.pytorch_cos_sim(emb1, emb2)
print(similarity.item())



model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
sentences = ["ok, ta liberado. segue em frente", "sim, voce pode fazer isso. esta correto"]
embeddings = model(sentences)

similarity = np.inner(embeddings[0], embeddings[1])
print(f'Similaridade: {similarity}')
