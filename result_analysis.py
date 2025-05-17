import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Definição das classes
class FreeFormAnswer:
    def __init__(self, free_form_answers, gold_answer=""):
        self.gold_answer = gold_answer
        # Garante que free_form_answers seja uma lista de strings
        if isinstance(free_form_answers, list):
            self.free_form_answers = [str(ans) for ans in free_form_answers if ans != ""]
        else:
            self.free_form_answers = [str(free_form_answers)] if free_form_answers != "" else []

    def to_dict(self):
        return {
            "gold_answer": self.gold_answer,
            "free_form_answers": self.free_form_answers
        }

class QuestionAnswer:
    def __init__(self, question, unanswerable, option_answers):
        self.question = question
        self.unanswerable = unanswerable
        if isinstance(option_answers, FreeFormAnswer):
            self.option_answers = option_answers
        else:
            self.option_answers = FreeFormAnswer(option_answers)

    def to_dict(self):
        return {
            "question": self.question,
            "unanswerable": self.unanswerable,
            "option_answers": self.option_answers.to_dict()
        }

class DatasetData:
    def __init__(self, title, abstract, full_text, qas, src_dataset):
        self.title = title
        self.abstract = abstract
        self.full_text = full_text
        self.qas = qas
        self.src_dataset = src_dataset

    def to_dict(self):
        return {
            "title": self.title,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "qas": [qa.to_dict() for qa in self.qas],
            "src_dataset": self.src_dataset
        }

class JoinedDataset:
    def __init__(self):
        self.dataset = []

    def to_dict(self):
        return {
            "dataset": [data.to_dict() for data in self.dataset]
        }

    def remove_full_text(self):
        for data in self.dataset:

            data.full_text = ""

class AiJuryModelOpinion:
    def __init__(self, model_name, opinion):
        self.is_correct = opinion["is_correct"]
        self.justification = opinion["justification"]

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "opinion": self.opinion
        }


class AiJury:
    def __init__(self, question, reference_answer, eval_answer):
        self.question = question
        self.reference_answer = reference_answer
        self.eval_answer = eval_answer
  
    def to_dict(self):
        return {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "eval_answer": self.eval_answer,
            "final_verdict": self.final_verdict,
            "models_opinion": [opinion.to_dict() for opinion in self.models_opinion]
        }

class LLMAnswer:
    def __init__(self, model_name, answer, option_answers_gt, question=None):
        self.question = question
        self.model_name = model_name
        self.answer = answer
        self.option_answers_gt = option_answers_gt
        self.similarity_score = None  # Similaridade geral (maior valor dentre as comparações)
        self.scores = []  # Lista de dicionários: { "gt_compared_answer": <gt>, "scores": <scores_dict> }
        logging.info(f"Instância de LLMAnswer criada para o modelo: {model_name} - Resposta: {answer}")

    def to_dict(self):
        # Transforma a lista de scores em um dicionário onde a chave é a resposta GT
        similarities = { item["gt_compared_answer"]: item["scores"] for item in self.scores }
        # check if ai jury results are empty - necessary because this was added after the first implementation
        ai_jury_results ={}
        for score in self.scores:
            #check if ai_jury_veredict_result key exists in score
            if not "ai_jury_veredict_result" in score:
                score["ai_jury_veredict_result"] = {}

            ai_jury_results[score["gt_compared_answer"]] = score["ai_jury_veredict_result"]
            
        return {
            "model_name": self.model_name,
            "answer": self.answer,
            "overall_similarity": self.similarity_score,
            "similarities": similarities,
            "ai_jury_results": ai_jury_results
        }

class ComparisonResult:
    def __init__(self, ground_truth_qas, text_title):
        self.question = ground_truth_qas.question
        self.text_title = text_title
        self.unanswerable = ground_truth_qas.unanswerable
        self.ground_truth = ground_truth_qas.option_answers.free_form_answers
        self.model_results = []  # Lista de instâncias de LLMAnswer

    def to_dict(self):
        model_results_list = [model_answer.to_dict() for model_answer in self.model_results]
        return {
            "question": self.question,
            "text_title": self.text_title,
            "unanswerable": self.unanswerable,
            "ground_truth": self.ground_truth,
            "model_results": model_results_list
        }

# Cria a pasta "analysis_results" se ela não existir
output_folder = "analysis_results"
os.makedirs(output_folder, exist_ok=True)

# Configuração para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# -------------------------------------------------------------------------------
# 1. Carregar os dados e converter objetos para dicionários, se necessário
# -------------------------------------------------------------------------------
with open('results/llm_generated_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Se data for um dicionário, converte para uma lista de seus valores
if isinstance(data, dict):
    data = list(data.values())

# Se os objetos possuem o método to_dict, converte todos para dicionários
if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'to_dict'):
    data = [comparison.to_dict() for comparison in data]

# -------------------------------------------------------------------------------
# 2. Identificar dinamicamente as chaves de score disponíveis
# -------------------------------------------------------------------------------
score_keys = set()
for comparison in data:
    for model in comparison['model_results']:
        # Para cada ground truth, verifica as chaves do dicionário de scores
        for gt_answer, scores in model['similarities'].items():
            if scores:  # só se o dicionário não estiver vazio
                score_keys.update(scores.keys())
score_keys = list(score_keys)
print("Chaves de score encontradas:", score_keys)

# -------------------------------------------------------------------------------
# 3. Construir listas para DataFrames: overall e detailed
# -------------------------------------------------------------------------------
overall_data = []   # Dados de similaridade geral (por modelo e questão)
detailed_data = []  # Dados detalhados (para cada comparação com ground truth)

for comparison in data:
    question = comparison['question']
    text_title = comparison['text_title']
    unanswerable = comparison.get('unanswerable', False)
    for model in comparison['model_results']:
        model_name = model['model_name']
        overall_similarity = model.get('overall_similarity', np.nan)
        
        overall_data.append({
            'question': question,
            'text_title': text_title,
            'model_name': model_name,
            'overall_similarity': overall_similarity,
            'unanswerable': unanswerable
        })
        
        # Para cada ground truth, pegar o dicionário de scores
        for gt_answer, scores in model['similarities'].items():
            row = {
                'question': question,
                'text_title': text_title,
                'model_name': model_name,
                'ground_truth': gt_answer
            }
            # Preenche para cada score key; se o dicionário estiver vazio, coloca NaN
            for key in score_keys:
                row[key] = scores.get(key, np.nan) if scores else np.nan
            detailed_data.append(row)

df_overall = pd.DataFrame(overall_data)
df_detailed = pd.DataFrame(detailed_data)

# -------------------------------------------------------------------------------
# 4. Cálculo da métrica média (avg_metric) e Estatísticas Descritivas dos Modelos (Overall Similarity)
# -------------------------------------------------------------------------------
df_detailed['avg_metric'] = df_detailed[score_keys].mean(axis=1, skipna=True)
print("Colunas em df_detailed:", df_detailed.columns)

overall_stats = df_overall.groupby('model_name')['overall_similarity'].agg(['mean', 'median', 'std', 'min', 'max'])
print("\nEstatísticas de Similaridade Geral por Modelo:")
print(overall_stats)
overall_stats.to_csv(os.path.join(output_folder, 'overall_stats.csv'))

# Gráfico de barras da média de overall_similarity por modelo
plt.figure()
sns.barplot(x='model_name', y='overall_similarity', data=df_overall, errorbar=None)
plt.xticks(rotation=45)
plt.title('Média de Overall Similarity por Modelo')
plt.ylabel('Overall Similarity')
plt.xlabel('Modelo')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'barplot_overall_similarity.png'))
plt.close()

# Histograma da distribuição de overall_similarity (todos os modelos)
plt.figure()
sns.histplot(df_overall['overall_similarity'].dropna(), bins=20, kde=True)
plt.title('Distribuição de Overall Similarity (Todos os Modelos)')
plt.xlabel('Overall Similarity')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'histogram_overall_similarity.png'))
plt.close()

# Boxplot de overall_similarity por modelo
plt.figure()
sns.boxplot(x='model_name', y='overall_similarity', data=df_overall)
plt.xticks(rotation=45)
plt.title('Boxplot de Overall Similarity por Modelo')
plt.xlabel('Modelo')
plt.ylabel('Overall Similarity')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'boxplot_overall_similarity.png'))
plt.close()

# -------------------------------------------------------------------------------
# 5. Análise de Correlação entre as Métricas de Similaridade (Score Keys)
# -------------------------------------------------------------------------------
corr_matrix = df_detailed[score_keys].corr()
print("\nMatriz de Correlação entre as Métricas de Similaridade:")
print(corr_matrix)

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlação entre Métricas de Similaridade")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'heatmap_score_keys_correlation.png'))
plt.close()

# -------------------------------------------------------------------------------
# 6. Correlação entre Overall Similarity e as Métricas Detalhadas
# -------------------------------------------------------------------------------
# Mescla os DataFrames usando left join para manter a coluna 'avg_metric'
df_merged = pd.merge(
    df_detailed,
    df_overall[['question', 'model_name', 'overall_similarity']],
    on=['question', 'model_name'],
    how='left'
)
print("Colunas em df_merged:", df_merged.columns)

corr_with_overall = df_merged[score_keys + ['overall_similarity']].corr()['overall_similarity'].drop('overall_similarity')
print("\nCorrelação de cada métrica com Overall Similarity:")
print(corr_with_overall)

# Scatter plots para cada métrica x overall_similarity
for key in score_keys:
    plt.figure()
    sns.scatterplot(x=key, y='overall_similarity', data=df_merged)
    plt.title(f'Overall Similarity vs {key}')
    plt.tight_layout()
    filename = f'scatter_overall_vs_{key.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

# -------------------------------------------------------------------------------
# 7. Análise de Variação entre Ground Truths para Cada Modelo e Questão
# -------------------------------------------------------------------------------
variation_data = df_detailed.groupby(['question', 'model_name'])['avg_metric'].agg(['min', 'max'])
variation_data['range'] = variation_data['max'] - variation_data['min']
print("\nVariação (range) das métricas médias por (questão, modelo):")
print(variation_data)
variation_data.to_csv(os.path.join(output_folder, 'variation_data.csv'))

# -------------------------------------------------------------------------------
# 8. Frequência de casos com dicionário de scores vazio
# -------------------------------------------------------------------------------
def is_empty_scores(row):
    return row[score_keys].isna().all()

df_detailed['empty_scores'] = df_detailed.apply(is_empty_scores, axis=1)
empty_counts = df_detailed.groupby('model_name')['empty_scores'].sum()
total_counts = df_detailed.groupby('model_name')['empty_scores'].count()
empty_percent = (empty_counts / total_counts * 100).round(2)
print("\nFrequência de casos com scores vazios por modelo:")
print(empty_counts)
print("Percentual de casos vazios por modelo:")
print(empty_percent)

pd.concat([empty_counts, empty_percent], axis=1, keys=['empty_counts', 'empty_percent'])\
  .to_csv(os.path.join(output_folder, 'empty_scores_frequency.csv'))

# -------------------------------------------------------------------------------
# 9. Análise de Casos “Unanswerable”
# -------------------------------------------------------------------------------
unanswerable_counts = df_overall['unanswerable'].value_counts()
print("\nContagem de casos unanswerable (nível de comparação):")
print(unanswerable_counts)

plt.figure()
sns.countplot(x='unanswerable', data=df_overall)
plt.title('Contagem de Casos Unanswerable')
plt.xlabel('Unanswerable')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'count_unanswerable.png'))
plt.close()

plt.figure()
sns.boxplot(x='unanswerable', y='overall_similarity', data=df_overall)
plt.title('Overall Similarity vs. Unanswerable')
plt.xlabel('Unanswerable')
plt.ylabel('Overall Similarity')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'boxplot_overall_vs_unanswerable.png'))
plt.close()

# -------------------------------------------------------------------------------
# 10. Correlação Inter-Modelos: Pivot table de overall_similarity por questão
# -------------------------------------------------------------------------------
pivot_overall = df_overall.pivot_table(index='question', columns='model_name', values='overall_similarity')
inter_model_corr = pivot_overall.corr()
print("\nCorrelação entre modelos (Overall Similarity):")
print(inter_model_corr)

plt.figure()
sns.heatmap(inter_model_corr, annot=True, cmap='coolwarm')
plt.title("Correlação Inter-Modelos de Overall Similarity")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'heatmap_inter_model_corr.png'))
plt.close()

# -------------------------------------------------------------------------------
# 11. Estatísticas Agregadas das Métricas Detalhadas por Modelo (geral)
# -------------------------------------------------------------------------------
aggregated_metrics = df_detailed.groupby('model_name')[score_keys].agg(['mean', 'median', 'std', 'min', 'max'])
print("\nEstatísticas agregadas das métricas detalhadas por modelo:")
print(aggregated_metrics)
aggregated_metrics.to_csv(os.path.join(output_folder, 'aggregated_metrics.csv'))

# -------------------------------------------------------------------------------
# Salvar um DataFrame para cada score separadamente em analysis_results/scores
# -------------------------------------------------------------------------------
scores_folder = os.path.join(output_folder, "scores")
os.makedirs(scores_folder, exist_ok=True)

for key in score_keys:
    df_score = df_detailed.groupby('model_name')[key].agg(['mean', 'median', 'std', 'min', 'max'])
    print(f"\nEstatísticas agregadas para o score: {key}")
    print(df_score)
    df_score.to_csv(os.path.join(scores_folder, f'aggregated_{key}.csv'))


# -------------------------------------------------------------------------------
# 12. Correlação entre a média dos scores (avg_metric) e Overall Similarity
# -------------------------------------------------------------------------------
corr_avg_overall = df_merged[['avg_metric', 'overall_similarity']].corr().iloc[0, 1]
print("\nCorrelação entre avg_metric e overall_similarity:")
print(corr_avg_overall)

plt.figure()
sns.scatterplot(x='avg_metric', y='overall_similarity', data=df_merged)
plt.title('Overall Similarity vs. Média dos Score Metrics (avg_metric)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'scatter_overall_vs_avg_metric.png'))
plt.close()

# -------------------------------------------------------------------------------
# 13. Exportar os DataFrames processados para CSV
# -------------------------------------------------------------------------------
df_overall.to_csv(os.path.join(output_folder, 'overall_similarity.csv'), index=False)
df_detailed.to_csv(os.path.join(output_folder, 'detailed_similarity.csv'), index=False)

print("Análises e imagens salvos na pasta:", output_folder)


# -------------------------------------------------------------------------------
# 14. Avaliação do AI Jury
# -------------------------------------------------------------------------------

# Listas auxiliares
jury_records = []          # veredicto final por ground-truth
jury_member_records = []   # votos de cada membro

for comparison in data:
    question = comparison['question']
    for model in comparison['model_results']:
        model_name = model['model_name']
        ai_jury_results = model.get('ai_jury_results', {})

        for gt_answer, verdict_dict in ai_jury_results.items():
            if not verdict_dict:
                continue

            final_verdict = verdict_dict.get('final_verdict')
            if final_verdict is None:
                continue

            # registro do veredito final (True = acerto)
            jury_records.append({
                'question'      : question,
                'model_name'    : model_name,
                'ground_truth'  : gt_answer,
                'final_verdict' : bool(final_verdict)
            })

            # agora percorre o dict de opiniões individuais
            for member_name, opinion_info in verdict_dict.get('models_opinion', {}).items():
                # opinion_info tem is_correct e justification
                member_vote = opinion_info.get('opinion')
                if member_vote is None:
                    continue
                jury_member_records.append({
                    'question'      : question,
                    'model_name'    : model_name,
                    'member'        : member_name,
                    'member_vote'   : bool(member_vote),
                    'correct'       : bool(member_vote) == bool(final_verdict)
                })

# -------------------------------------------------------------------------------
# 15. DataFrames do júri
# -------------------------------------------------------------------------------
df_jury   = pd.DataFrame(jury_records)
df_member = pd.DataFrame(jury_member_records)

df_jury.to_csv(os.path.join(output_folder, 'jury_results.csv'), index=False)
df_member.to_csv(os.path.join(output_folder, 'jury_member_votes.csv'), index=False)

# -------------------------------------------------------------------------------
# 16. Precisão, acertos e erros por modelo
# -------------------------------------------------------------------------------
precision_per_model = df_jury.groupby('model_name')['final_verdict'].mean().sort_values(ascending=False)
counts_per_model    = df_jury.groupby(['model_name', 'final_verdict']).size().unstack(fill_value=0)

precision_per_model.to_csv(os.path.join(output_folder, 'precision_per_model.csv'))
counts_per_model.to_csv(os.path.join(output_folder, 'correct_incorrect_counts.csv'))

# Gráfico – precisão
plt.figure()
sns.barplot(x=precision_per_model.index, y=precision_per_model.values, errorbar=None)
plt.title('Precisão do AI Jury por Modelo')
plt.ylabel('Precisão')
plt.xlabel('Modelo')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'barplot_precision_per_model.png'))
plt.close()

# Gráfico – acertos e erros
counts_per_model.plot(kind='bar', stacked=False, figsize=(10,6))
plt.title('Contagem de Acertos (True) e Erros (False) por Modelo')
plt.ylabel('Quantidade')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'barplot_correct_incorrect_per_model.png'))
plt.close()

# -------------------------------------------------------------------------------
# 17. Comparação entre veredicto do júri e métricas BERT / SBERT
# -------------------------------------------------------------------------------
metric_cols = [c for c in score_keys if 'bert' in c.lower() or 'sbert' in c.lower()]
if metric_cols:
    # junta verdictos com métricas
    df_compare = pd.merge(
        df_detailed,                   # contém métricas
        df_jury[['question', 'model_name', 'final_verdict']],
        on=['question', 'model_name'],
        how='inner'
    )
    # correlação verdicto (0-1) x métricas
    corr_vs_metrics = df_compare[metric_cols + ['final_verdict']].corr()['final_verdict'].drop('final_verdict')
    corr_vs_metrics.to_csv(os.path.join(output_folder, 'corr_verdict_vs_metrics.csv'))

    # boxplots métricas x veredicto
    for col in metric_cols:
        plt.figure()
        sns.boxplot(x='final_verdict', y=col, data=df_compare)
        plt.title(f'{col} vs Veredicto do Júri')
        plt.xlabel('Veredicto do Júri (0 = erro, 1 = acerto)')
        plt.ylabel(col)
        plt.tight_layout()
        fname = f'boxplot_{col.replace(" ", "_")}_vs_verdict.png'
        plt.savefig(os.path.join(output_folder, fname))
        plt.close()
else:
    logging.warning("Nenhuma coluna de métrica BERT / SBERT encontrada em score_keys.")

# -------------------------------------------------------------------------------
# 18. Votos por membro do júri e sua precisão
# -------------------------------------------------------------------------------
if not df_member.empty:
    # barras True / False por membro e modelo
    member_count = df_member.groupby(['model_name', 'member', 'member_vote']).size().unstack(fill_value=0)
    member_prec  = df_member.groupby('member')['correct'].mean().sort_values(ascending=False)

    member_count.to_csv(os.path.join(output_folder, 'member_vote_counts.csv'))
    member_prec.to_csv(os.path.join(output_folder, 'member_precision.csv'))

    # gráfico por modelo mostrando barra para cada membro (True / False)
    for model_name, subdf in member_count.groupby(level=0):
        subdf = subdf.droplevel(0)  # remove o índice do modelo
        subdf.plot(kind='bar', stacked=False, figsize=(10,6))
        plt.title(f'Votos True / False por Membro – {model_name}')
        plt.ylabel('Quantidade')
        plt.xlabel('Membro do Júri')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fname = f'barplot_member_votes_{model_name}.png'.replace('/', '_')
        plt.savefig(os.path.join(output_folder, fname))
        plt.close()

    # gráfico de precisão por membro
    plt.figure()
    sns.barplot(x=member_prec.index, y=member_prec.values, errorbar=None)
    plt.title('Precisão Individual dos Membros do Júri')
    plt.ylabel('Precisão')
    plt.xlabel('Membro')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'barplot_member_precision.png'))
    plt.close()
else:
    logging.warning("df_member vazio – não foi possível gerar gráficos de membros do júri.")

print("Novas avaliações do AI Jury finalizadas – resultados adicionados em", output_folder)
