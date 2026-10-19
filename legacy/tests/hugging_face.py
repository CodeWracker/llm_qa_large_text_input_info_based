from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["Pergunta: Quantas rodas tem um carro?\nReposta referencia: O carro tem 4 rodas.\nResposta aluno:4\njson de avaliação tem uma key chamada is_correct que é booleana. aqui esta o JSON: {is_correct:"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
