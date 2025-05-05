"""
AI Jury - sends prompts to local Ollama models and aggregates their verdicts.
"""

import json
import logging
import os
import time
from pprint import pprint
from typing import Dict, List

import requests


# ----------------------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------------------

def build_prompt(question: str, reference: str, eval_answer: str) -> str:
    """
    Create the prompt sent to each model.
    The JSON schemas below must not be changed.
    """
    correct_schema = '{"justification": "justification example", "is_correct": true}'
    incorrect_schema = '{"justification": "justification example", "is_correct": false}'

    return f"""
You are an impartial AI jury. Your only task is to decide whether the evaluate's answer (EVAL) is factually **correct** when compared with the authoritative answer (REFERENCE) for the given question (Q).

Definition of **correct**:
- Every factual statement in EVAL must match the content of REFERENCE.
- Paraphrases and synonyms are fine, but any contradiction, omission, or extra fact that changes the meaning makes it **incorrect**.
- Superficial or lexical similarity alone is **not** enough.
- IMPORTANT: If REFERENCE == "N/A" and EVAL == "N/A" → is_correct MUST be true.


Output format:
- Exactly one ```json``` code block.
- Keys: "is_correct" (boolean) and "justification" (string).
- The value of "is_correct" must align with your justification. If justification says the answer is wrong, "is_correct" must be false.

Examples of the required JSON (do not copy literally, they are schematic):

Correct answer:
```json
{correct_schema}
````

Incorrect answer:

```json
{incorrect_schema}
```

Guidelines:

1. Read Q, REFERENCE and EVAL carefully.use no external knowledge.
2. Decide:
    - If EVAL states the *same facts* as REFERENCE → "is_correct": true
    - Otherwise → "is_correct": false
3. Keep the justification brief.
4. No text is allowed outside the JSON block.
5. justification must be in English, even if the question is in another language.
6. justification must not contain any code or JSON.
7. justification must not be empty.
8. Justificatioon must refer to the reference answer when explaining why the eval answer is correct or incorrect.
9. is_correct must be a boolean value, not a string.
10. is_correct and justification must align with each other.

Q: {question}
REFERENCE: {reference}
EVAL: {eval_answer}
"""

# ----------------------------------------------------------------------
# Model interaction
# ----------------------------------------------------------------------

def query_model(model: str, prompt: str) -> Dict:
    """
    Send the prompt to a model until a valid JSON answer is returned.
    """
    ollama_url = f"{os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    attempt = 1
    while True:
        try:
            resp = requests.post(
                ollama_url,
                headers=headers,
                data=json.dumps(payload),
            )
            resp.raise_for_status()
            content = resp.json()["response"]
            verdict = json.loads(content)
            # checks if the respose has valid JSON format
            if "is_correct" not in verdict or "justification" not in verdict:
                raise ValueError("ai_jury.py - Invalid JSON format. Missing keys.")
            if not isinstance(verdict["is_correct"], bool):
                raise ValueError("ai_jury.py - Invalid JSON format. 'is_correct' must be a boolean.")
            if not isinstance(verdict["justification"], str):
                raise ValueError("ai_jury.py - Invalid JSON format. 'justification' must be a string.")
            if len(verdict["justification"]) == 0:
                raise ValueError("ai_jury.py - Invalid JSON format. 'justification' must not be empty.")
            # checks if the response is a valid JSON
            logging.debug("Model %s answered on attempt %d", model, attempt)
            return verdict  # {"is_correct": bool, "justification": str}
        except Exception as exc:
            logging.warning("AI Jury - Attempt %d failed for model %s: %s", attempt, model, exc)
            attempt += 1

# ----------------------------------------------------------------------
# Verdict aggregation
# ----------------------------------------------------------------------

def aggregate_verdicts(verdicts: Dict[str, Dict]) -> bool:
    """
    Simple majority vote.
    """
    positives = sum(1 for v in verdicts.values() if v["is_correct"])
    return positives / len(verdicts) >= 0.5

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def ask_ai_jury(question: str, reference: str, eval_answer: str) -> Dict:
    """
    Query all models and return their opinions plus the final verdict.
    """
    models: List[str] = [
        "deepseek-r1:1.5b",
        "llama3.2:1b",
        "gemma3:1b",
    ]
    prompt = build_prompt(question, reference, eval_answer)
    models_opinion: Dict[str, Dict] = {}
    logging.info(f"AI Jury - Asking models to evaluate the correctness of the answer. Models Council: {models}")
    for model in models:
        logging.debug("AI Jury - Querying model: %s", model)
        models_opinion[model] = query_model(model, prompt)

    final_verdict = aggregate_verdicts(models_opinion)
    return {
        "models_opinion": models_opinion,
        "final_verdict": final_verdict,
    }

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Logging setup
    # ----------------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    question = "What is the capital of France?"
    reference_answer = "The capital of France is Paris."
    eval_answer = "The capital of France is Berlin."

    print(f"Question: {question}")
    print(f"Reference Answer: {reference_answer}")
    print(f"Eval Answer: {eval_answer}")
    print("Asking AI Jury...")

    result = ask_ai_jury(question, reference_answer, eval_answer)
    pprint(result)

    
    question = "Does the paper report macro F1?"
    reference_answer = "yes"
    eval_answer = "N/A"
    print("n\n\n\n")
    print(f"Question: {question}")
    print(f"Reference Answer: {reference_answer}")
    print(f"Eval Answer: {eval_answer}")
    print("Asking AI Jury...")

    result = ask_ai_jury(question, reference_answer, eval_answer)
    pprint(result)