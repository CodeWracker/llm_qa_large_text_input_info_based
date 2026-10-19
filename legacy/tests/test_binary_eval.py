import os
import re
import json
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from nltk.stem import PorterStemmer

from huggingface_hub import InferenceApi

# ------------------------
# 0) Configuração inicial
# ------------------------
# Exige que você tenha exportado seu token HF em HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("test_binary_eval.py - Defina a variável de ambiente HF_TOKEN com seu token Hugging Face")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# 1) Carrega avaliadores clássicos
# ------------------------
print("Carregando SBERT e NLI…")
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
nli_pipe    = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    device=0 if device.type=="cuda" else -1
)

stemmer = PorterStemmer()

# ------------------------
# 2) Inicializa Inference API para LLMs
# ------------------------
print("Inicializando Hugging Face Inference API para Mistral, DeepSeek e Qwen…")
mistral_api = InferenceApi(repo_id="mistralai/Mistral-7B-v0.1", token=HF_TOKEN)
deepseek_api = InferenceApi(repo_id="deepseek/DeepSeek-3B",   token=HF_TOKEN)
qwen_api     = InferenceApi(repo_id="qwen/Qwen-2B",          token=HF_TOKEN)

# ------------------------
# 3) Funções auxiliares
# ------------------------
def preprocess(text: str) -> set[str]:
    toks = re.findall(r'\b\w+\b', text.lower())
    return {stemmer.stem(t) for t in toks if t not in stop_words and len(t) > 1}

def keyword_matching_score(ref: str, resp: str) -> float:
    tref = preprocess(ref)
    tres = preprocess(resp)
    return 0.0 if not tref else len(tref & tres) / len(tref)

def avaliador_lexico(resp: str) -> bool:
    low = resp.lower()
    return ("roda" in low) and ("4" in low or "quatro" in low)

def avaliador_curto_contextual(perg: str, resp: str, ref: str) -> bool:
    toks = re.findall(r'\b\w+\b', resp.lower())
    if len(toks) <= 2:
        nr = re.findall(r'\b\d+(?:\.\d+)?\b', ref)
        na = re.findall(r'\b\d+(?:\.\d+)?\b', resp)
        return bool(nr and na and nr[0] == na[0])
    return False

def llm_vote(name: str, api: InferenceApi, prompt: str) -> tuple[bool, str]:
    """
    Envia prompt para a InferenceApi e retorna (is_correct, raw_output).
    Espera que o modelo responda com um JSON {"is_correct": true/false}.
    """
    out = api(inputs=prompt)
    # Alguns endpoints retornam dicts, outros strings
    text = out if isinstance(out, str) else out.get("generated_text", str(out))
    text = text.strip()
    # Extrai JSON
    start, end = text.find("{"), text.rfind("}")
    jstr = text[start:end+1] if (start!=-1 and end!=-1) else text
    try:
        data = json.loads(jstr)
        vote = bool(data.get("is_correct", False))
    except json.JSONDecodeError:
        vote = False
    return vote, jstr

# ------------------------
# 4) Pipeline principal
# ------------------------
def main():
    pergunta       = "Quantas rodas tem um carro?"
    referencia     = "O carro tem 4 rodas"
    resposta_aluno = input("Resposta do aluno: ").strip()

    print(f"\nPergunta:               {pergunta}")
    print(f"Resposta de referência: {referencia}")
    print(f"Resposta do aluno:      {resposta_aluno}\n")

    votos: list[bool] = []

    # 4.1 SBERT
    emb_a  = sbert_model.encode(resposta_aluno,   convert_to_tensor=True).to(device)
    emb_r  = sbert_model.encode(referencia,       convert_to_tensor=True).to(device)
    sim    = float(util.cos_sim(emb_a, emb_r)[0])
    vote_s = sim >= 0.75
    votos.append(vote_s)
    print(f"[SBERT] Similaridade: {sim:.2f} → {'CORRETO' if vote_s else 'INCORRETO'}")

    # 4.2 NLI
    nli_in   = f"{referencia} </s> {resposta_aluno}"
    out_nli  = nli_pipe(nli_in)[0]
    vote_nli = (out_nli['label']=="ENTAILMENT") and (out_nli['score']>0.8)
    votos.append(vote_nli)
    print(f"[NLI] {out_nli['label']} (conf: {out_nli['score']:.2f}) → {'CORRETO' if vote_nli else 'INCORRETO'}")

    # 4.3 Heurística lexical
    vote_lex = avaliador_lexico(resposta_aluno)
    votos.append(vote_lex)
    print(f"[LEXICAL] Presença de ‘roda’+‘4/quatro’ → {'CORRETO' if vote_lex else 'INCORRETO'}")

    # 4.4 Keyword Matching
    kw_score = keyword_matching_score(referencia, resposta_aluno)
    vote_kw  = kw_score >= 0.5
    votos.append(vote_kw)
    print(f"[KEYWORDS] Score: {kw_score:.2f} → {'CORRETO' if vote_kw else 'INCORRETO'}")

    # 4.5 Contextual curto
    vote_ctx = avaliador_curto_contextual(pergunta, resposta_aluno, referencia)
    votos.append(vote_ctx)
    print(f"[CONTEXTUAL] Curta + número correto → {'CORRETO' if vote_ctx else 'INCORRETO'}")

    # 4.6 Juízes LLM via API
    prompt = (
        f"Pergunta: {pergunta}\n"
        f"Resposta de referência: {referencia}\n"
        f"Resposta do aluno: {resposta_aluno}\n\n"
        "Responda **APENAS** com um JSON no formato {\"is_correct\": true} ou {\"is_correct\": false}."
    )
    for name, api in [
        ("mistral-7b", mistral_api),
        ("deepseek-3b", deepseek_api),
        ("qwen-2b",     qwen_api)
    ]:
        vote_llm, raw = llm_vote(name, api, prompt)
        votos.append(vote_llm)
        print(f"[LLM:{name}] {raw} → {'CORRETO' if vote_llm else 'INCORRETO'}")

    # 4.7 Decisão final
    total    = len(votos)
    positive = sum(votos)
    needed   = total // 2 + 1
    print(f"\nTotal de votos positivos: {positive}/{total}")
    final = "CORRETA" if positive >= needed else "INCORRETA"
    print(f">> RESPOSTA FINAL: {final}")

if __name__ == "__main__":
    main()
