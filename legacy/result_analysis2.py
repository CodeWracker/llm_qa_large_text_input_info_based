import os
import pickle
import pandas as pd
import numpy as np
import logging
from pprint import pprint
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
output_folder = "analysis_results2"
os.makedirs(output_folder, exist_ok=True)




with open('results/llm_generated_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Se data for um dicionário, converte para uma lista de seus valores
if isinstance(data, dict):
    data = list(data.values())

# Se os objetos possuem o método to_dict, converte todos para dicionários
if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'to_dict'):
    data = [comparison.to_dict() for comparison in data]


score_keys = set()
for comparison in data:
    for model in comparison['model_results']:
        # Para cada ground truth, verifica as chaves do dicionário de scores
        for gt_answer, scores in model['similarities'].items():
            if scores:  # só se o dicionário não estiver vazio
                score_keys.update(scores.keys())
score_keys = list(score_keys)
print("Chaves de score encontradas:", score_keys)


models_evaluated = []
for comparison in data:
    for model in comparison['model_results']:
        if model['model_name'] not in models_evaluated:
            models_evaluated.append(model['model_name'])
models_evaluated_df = pd.DataFrame(models_evaluated, columns=['model_name'])
models_evaluated_df.to_csv(os.path.join(output_folder, 'models_evaluated.csv'), index=False)

models_answers = []
jury_models_veredicts = []
similarities_data = []
ids = []
idx = 0
for comparison in data:
    for model in comparison['model_results']:
        idx += 1
        id = idx
        question = comparison['question']
        unanswerable = comparison['unanswerable']
        text_title = comparison['text_title']
        
        
        model_name = model['model_name']
        answer = model['answer']
        overall_similarity = model['overall_similarity']
        
        ai_jury_results = model['ai_jury_results']
        finally_verdict = False
        jury_ans_id = 0
        for ai_jury_result in ai_jury_results:
            if(ai_jury_results[ai_jury_result] == {}):
                continue
            jury_ans_id += 1
            models = ai_jury_results[ai_jury_result]['models_opinion'].keys()
            final_verdict_gt = ai_jury_results[ai_jury_result]['final_verdict']
            if final_verdict_gt:
                finally_verdict = True
            for jury_model in ai_jury_results[ai_jury_result]['models_opinion']:
                jury_models_veredicts.append({
                    'id': str(id) + '-' + str(jury_ans_id),
                    'response_id': id,
                    'jury_model_name': jury_model,
                    'answer_model_name': model_name,
                    'is_correct': ai_jury_results[ai_jury_result]['models_opinion'][jury_model]['is_correct'],
                })
                
        similarities = model['similarities']
        sim_questions = similarities.keys()
        sim_ans_id = 0
        for sim_question in sim_questions:
            sim_ans_id += 1
            scores = similarities[sim_question].keys()
            for score_name in scores:
                similarities_data.append({
                    'id': str(id) + '-' + str(sim_ans_id),
                    'response_id': id,
                    'model_name': model_name,
                    'score_name': score_name,
                    'score_value': similarities[sim_question][score_name]
                })
        
            
        data = {
            'id': id,
            'model_name': model_name,
            'unanswerable': unanswerable,
            'text_title': text_title,
            'jury_final_verdict': finally_verdict,
            'overall_similarity': overall_similarity,
        }
        models_answers.append(data)
        
models_answers_df = pd.DataFrame(models_answers)
models_answers_df.to_csv(os.path.join(output_folder, 'models_answers.csv'), index=False)

jury_models_veredicts_df = pd.DataFrame(jury_models_veredicts)
jury_models_veredicts_df.to_csv(os.path.join(output_folder, 'jury_models_veredicts.csv'), index=False)

similarities_df = pd.DataFrame(similarities_data)
similarities_df.to_csv(os.path.join(output_folder, 'similarities.csv'), index=False)


# analises

import seaborn as sns
import matplotlib.pyplot as plt

# Configuração para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Boxplot de overall similarity por jury_final_verdict, separado por modelo
fig, ax = plt.subplots(figsize=(20, 12))
sns.boxplot(
    data=models_answers_df,
    x='jury_final_verdict',
    y='overall_similarity',
    hue='model_name',
    ax=ax
)
ax.set_title('Similaridade Geral por Veredito Final do Júri, por Modelo')
ax.set_xlabel('Veredito Final do Júri')
ax.set_ylabel('Similaridade Geral')

# posiciona legenda fora à direita
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title='Modelo'
)

# ajusta layout deixando espaço para a legenda
fig.tight_layout(rect=[0, 0, 0.85, 1])

# salva
boxplot_path = os.path.join(output_folder, 'boxplot_overall_similarity_jury_verdict.png')
fig.savefig(boxplot_path)
plt.close(fig)
print(f"Boxplot salvo em: {boxplot_path}")

# Barplot da média da similarity por jury_final_verdict, separado por modelo
fig2, ax2 = plt.subplots(figsize=(20, 12))
sns.barplot(
    data=models_answers_df,
    x='jury_final_verdict',
    y='overall_similarity',
    hue='model_name',
    estimator=np.mean,
    errorbar=None,
    ax=ax2
)
ax2.set_title('Média da Similaridade Geral por Veredito Final do Júri, por Modelo')
ax2.set_xlabel('Veredito Final do Júri')
ax2.set_ylabel('Média da Similaridade Geral')

ax2.legend(
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title='Modelo'
)

fig2.tight_layout(rect=[0, 0, 0.8, 1])
barplot_path = os.path.join(output_folder, 'barplot_mean_overall_similarity_jury_verdict.png')
fig2.savefig(barplot_path, bbox_inches='tight')
plt.close(fig2)
print(f"Barplot salvo em: {barplot_path}")




import scipy.stats as stats
from pandas.plotting import scatter_matrix
# 0) renomeia pra facilitar o merge
models_answers_df.rename(columns={'id':'response_id'}, inplace=True)

# 1) Prepara data wide de todas as métricas de score
score_metrics = similarities_df['score_name'].unique().tolist()
scores_wide = similarities_df.pivot_table(
    index=['response_id','model_name'],
    columns='score_name',
    values='score_value'
).reset_index()

# 2) merge com overall_similarity e verdict
merged = pd.merge(
    scores_wide,
    models_answers_df[['response_id','model_name','overall_similarity','jury_final_verdict']],
    on=['response_id','model_name']
)

# 3) Scatter‐matrix entre métricas
sm = scatter_matrix(
    merged[score_metrics],
    figsize=(12,12),
    diagonal='kde',
    alpha=0.5
)
plt.suptitle('Scatter Matrix entre Métricas de Score', y=0.92)
plt.tight_layout()
plt.savefig(os.path.join(output_folder,'scatter_matrix_scores.png'), bbox_inches='tight')
plt.close('all')

# 4) Heatmap de correlação
corr = merged[score_metrics].corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Heatmap de Correlação entre Métricas de Score')
fig.tight_layout()
fig.savefig(os.path.join(output_folder,'heatmap_scores_correlation.png'), bbox_inches='tight')
plt.close(fig)

# 5) p‐valores (Pearson + t‐test)
pvals = []
for m in score_metrics:
    x = merged[m].dropna()
    y = merged['overall_similarity'].dropna()
    corr_coef, p_corr = stats.pearsonr(x, y)
    grp0 = merged.loc[merged['jury_final_verdict']==False, m].dropna()
    grp1 = merged.loc[merged['jury_final_verdict']==True,  m].dropna()
    t_stat, p_ttest = stats.ttest_ind(grp0, grp1, nan_policy='omit')
    pvals.append({
        'metric': m,
        'pearson_corr': corr_coef,
        'pearson_pval': p_corr,
        'ttest_stat': t_stat,
        'ttest_pval': p_ttest
    })
pd.DataFrame(pvals).to_csv(os.path.join(output_folder,'pvalues_scores.csv'), index=False)

# 6) Precisão do veredito final por modelo
prec = models_answers_df.groupby('model_name')['jury_final_verdict'].mean().reset_index()
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(data=prec, x='model_name', y='jury_final_verdict', ax=ax)
ax.set_title('Precisão do Veredito Final por Modelo')
ax.set_xlabel('Modelo')
ax.set_ylabel('Precisão (True / Total)')
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(os.path.join(output_folder,'precision_by_model.png'), bbox_inches='tight')
plt.close(fig)

# 7) Barplot e boxplot de cada métrica + overall_similarity por modelo
all_metrics = score_metrics + ['overall_similarity']
for metric in all_metrics:
    df = merged if metric in score_metrics else models_answers_df

    # barplot
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(
        data=df,
        x='model_name',
        y=metric,
        estimator=np.mean,
        errorbar=None,
        ax=ax
    )
    ax.set_title(f'Média de {metric} por Modelo')
    ax.set_xlabel('Modelo')
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f'barplot_{metric}_by_model.png'), bbox_inches='tight')
    plt.close(fig)

    # boxplot
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(
        data=df,
        x='model_name',
        y=metric,
        ax=ax
    )
    ax.set_title(f'Distribuição de {metric} por Modelo')
    ax.set_xlabel('Modelo')
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f'boxplot_{metric}_by_model.png'), bbox_inches='tight')
    plt.close(fig)

# 8) Contagem True/False por jury_model para cada modelo
counts = jury_models_veredicts_df.groupby(
    ['answer_model_name','jury_model_name','is_correct']
).size().reset_index(name='count')
counts.to_csv(os.path.join(output_folder,'jury_model_counts.csv'), index=False)

g = sns.catplot(
    data=counts,
    x='answer_model_name',
    y='count',
    hue='jury_model_name',
    col='is_correct',
    kind='bar',
    height=6,
    aspect=1.5
)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Contagem de Decisões por Jury Model e Modelo')
g.savefig(os.path.join(output_folder,'jury_model_counts_by_is_correct.png'), bbox_inches='tight')
plt.close('all')

# 9) Unanimidade do júri por modelo
um = jury_models_veredicts_df.groupby(
    ['response_id','answer_model_name']
)['is_correct'].nunique().reset_index(name='n_unique')
um['unanimous'] = um['n_unique']==1
unanimity = um.groupby('answer_model_name')['unanimous'].mean().reset_index()
unanimity.to_csv(os.path.join(output_folder,'unanimity_summary.csv'), index=False)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(
    data=unanimity,
    x='answer_model_name',
    y='unanimous',
    ax=ax
)
ax.set_title('Proporção de Unanimidade do Júri por Modelo')
ax.set_xlabel('Modelo')
ax.set_ylabel('Proporção de Unanimidade')
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(os.path.join(output_folder,'unanimity_by_model.png'), bbox_inches='tight')
plt.close(fig)