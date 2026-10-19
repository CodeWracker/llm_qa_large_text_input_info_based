# Use uma imagem base com Python 3.10
FROM python:3.10-slim

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Atualiza o pip e instala as dependências Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Pré-baixa recursos do NLTK, Glove e modelo do Transformers
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
RUN python -c "from gensim import downloader; downloader.load('glove-wiki-gigaword-300', return_path=True)"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')"

# Comando padrão para iniciar o container (ajuste conforme necessário)
CMD ["bash"]
