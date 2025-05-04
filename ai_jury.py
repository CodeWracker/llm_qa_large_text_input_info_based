"""
send to ollama server and get response within a ```json``` code block
- name: ai_jury

"""

import requests
import json
import logging
from pprint import pprint

ollama_url = "http://localhost:11434/api/generate"




def get_propmpt(question, reference_answer, eval_answer):
    correct_json = '{"is_correct": true, "justification": "write the justification here"}'
    incorrect_json = '{"is_correct": false, "justification": "write the justification here"}'
    prompt = f"""You are an AI jury. You will be given a question, a reference answer, and an evaluation answer. Your task is to evaluate the evaluation answer based on the reference answer and the question ONLTY. do not use prior information, just use the question, reference answer, and evaluation answer to evaluate the similarity.
    return prompt
    
    Q: {question}
    REFERENCE: {reference_answer}
    EVAL: {eval_answer}
    
    is the eval answer correct? answer with a json with a boolean key "is_correct" only, within a ```, nothing more. no explanation before or after. jsut de code block with the json as the following example

    DO NOT USE KNOLEGE OUTSIDE THE BOUNDS OF THE REFERENCES ABOVE. USE ONLY THE SIMILARITY TO THE REFERENCE ANSWER TO DETERMINE THE CORRECTNESS OF THE STUDENT ANSWER.

    for correct answers
    ```json
    {correct_json}
    ```

    for false answers
    ```json
    {incorrect_json}
    ```
    """
    return prompt


def ask_ai_jury(question, reference_answer, eval_answer):
    prompt = get_propmpt(question, reference_answer, eval_answer)
    
    veredict = {
        "models_opinion": {},
    }
    models = [
        "deepseek-r1:1.5b",
        "llama3.2:1b",
        "gemma3:1b"
    ]
    for model in models:
        logging.info(f"Model: {model}")
        success = False
        model_prompt = prompt
        while not success:
            try:
                payload = {
                    'model': model,
                    "prompt": model_prompt,
                    "format": "json",
                    "stream": False,
                }
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                response = requests.post(ollama_url, headers=headers, data=json.dumps(payload))
                if response.status_code != 200:
                    logging.error(f"Error: {response.status_code} - {response.text}")
                    continue
                
                model_opinion = json.loads(json.loads(response.text)['response'])
                
                logging.info(f"Model: {model} - Veredict: {model_opinion['is_correct']}")
                logging.info(f"Model: {model} - Justification: {model_opinion['justification']}")
                veredict['models_opinion'][model] = model_opinion
                
                success = True
            except Exception as e:
                logging.error(f"Error while processing model {model}: {e}")
                # passing the error to the next iteration
                model_prompt = prompt + f"\nPrevious Error: {e}"
                continue
    
    true_counter = 0
    for model in models:
        print(f"Model: {model} - Veredict: {veredict['models_opinion'][model]['is_correct']}")
        if veredict['models_opinion'][model]['is_correct']:
            true_counter += 1
    
    veredict['final_verdict'] = true_counter / len(models) >= 0.5
    
    return veredict


if __name__ == "__main__":
    question = "What is the capital of France?"
    reference_answer = "The capital of France is Paris."
    eval_answer = "The capital of France is Berlin."
    
    print(f'Question: {question}')
    print(f'Reference Answer: {reference_answer}')
    print(f'Eval Answer: {eval_answer}')
    print('Asking AI Jury...')
    
    result = ask_ai_jury(question, reference_answer, eval_answer)
    pprint(result)