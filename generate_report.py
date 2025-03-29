import os
import pandas as pd
from pylatex import Document, Section, Subsection, Command, Figure, NoEscape, Package

# ============================================================================
# 1. Carregar os CSVs gerados anteriormente
# ============================================================================
csv_folder = "analysis_results"
overall_stats_df = pd.read_csv(os.path.join(csv_folder, "overall_stats.csv"), index_col=0)
empty_scores_df = pd.read_csv(os.path.join(csv_folder, "empty_scores_frequency.csv"), index_col=0)
aggregated_metrics_df = pd.read_csv(os.path.join(csv_folder, "aggregated_metrics.csv"), index_col=0)

# ---------------------------------------------------------------------------
# Função para escapar manualmente os underscores em colunas e índice
# ---------------------------------------------------------------------------
def escape_df(df):
    # Use raw strings para evitar problemas com barras invertidas
    df.index = df.index.to_series().str.replace('_', r'\_', regex=False)
    df.columns = df.columns.str.replace('_', r'\_', regex=False)
    return df

overall_stats_df = escape_df(overall_stats_df)
empty_scores_df = escape_df(empty_scores_df)
aggregated_metrics_df = escape_df(aggregated_metrics_df)

# ---------------------------------------------------------------------------
# Converter os DataFrames para tabelas em LaTeX com escape=False, 
# pois os underscores já foram escapados manualmente
# ---------------------------------------------------------------------------
overall_stats_table = overall_stats_df.to_latex(escape=False)
empty_scores_table = empty_scores_df.to_latex(escape=False)
aggregated_metrics_table = aggregated_metrics_df.to_latex(escape=False)

# ============================================================================
# 2. Gerar análises simples para as seções de Discussão e Conclusões
# ============================================================================
# Identificar o modelo com maior e menor média de overall similarity
max_mean_model = overall_stats_df['mean'].idxmax()
max_mean_value = overall_stats_df.loc[max_mean_model, 'mean']
min_mean_model = overall_stats_df['mean'].idxmin()
min_mean_value = overall_stats_df.loc[min_mean_model, 'mean']

# Identificar qual modelo possui o maior percentual de casos vazios
# Note que aqui a coluna já foi escapada, então use 'empty\_percent'
max_empty_model = empty_scores_df['empty\\_percent'].idxmax()
max_empty_percent = empty_scores_df.loc[max_empty_model, 'empty\\_percent']

# Texto da discussão baseado nos dados
discussion_text = NoEscape(rf"""
A análise dos dados indica que o modelo \textbf{{{max_mean_model}}} apresentou a maior média de overall similarity ({max_mean_value:.2f}), 
enquanto o modelo \textbf{{{min_mean_model}}} apresentou a menor média ({min_mean_value:.2f}). Essa diferença ressalta variações na performance 
dos modelos em aderência às respostas de referência.

Adicionalmente, observa-se que o percentual de casos com dicionário de scores vazio foi mais elevado para o modelo 
\textbf{{{max_empty_model}}} ({max_empty_percent}%), sugerindo que podem haver dificuldades na geração ou na coleta dos scores para esse modelo.

A análise das estatísticas agregadas das métricas detalhadas evidencia variações na consistência dos modelos, o que pode ser explorado 
para identificar possíveis ajustes nos algoritmos de resposta.
""")

# Texto das conclusões
conclusion_text = NoEscape(r"""
Com base nos dados analisados, pode-se concluir que:
\begin{itemize}
    \item Modelos com maiores médias de overall similarity tendem a aderir melhor às respostas de referência, embora a variabilidade dos scores deva ser considerada.
    \item O elevado percentual de casos com scores vazios em alguns modelos indica a necessidade de investigar a qualidade dos dados e o processo de geração de scores.
    \item As estatísticas agregadas das métricas detalhadas oferecem insights sobre a consistência dos modelos, o que pode orientar ajustes para aprimorar a performance.
\end{itemize}
""")

# ============================================================================
# 3. Criação do documento LaTeX com inclusão das tabelas e textos dinâmicos
# ============================================================================
geometry_options = {
    "tmargin": "2cm",
    "lmargin": "2cm",
    "rmargin": "2cm",
    "bmargin": "2cm"
}
doc = Document("RelatorioAnalise", geometry_options=geometry_options)

# Adicionar os pacotes necessários
doc.packages.append(Package("graphicx"))
doc.packages.append(Package("float"))
doc.packages.append(Package("booktabs"))

# Pré-ambiente: título, autor, data
doc.preamble.append(Command("title", "Relatório de Análise de Similaridade entre Modelos"))
doc.preamble.append(Command("author", "Seu Nome"))
doc.preamble.append(Command("date", NoEscape(r"\today")))
doc.append(NoEscape(r"\maketitle"))
doc.append(NoEscape(r"\tableofcontents"))
doc.append(NoEscape(r"\newpage"))

# ---------------------------------------------------------------------------
# Seção 1. Introdução
# ---------------------------------------------------------------------------
with doc.create(Section("Introdução", numbering=False)):
    doc.append(NoEscape(
        "Este relatório apresenta a análise comparativa dos modelos baseados nas métricas de similaridade entre respostas. "
        "Os dados foram processados a partir de um arquivo pickle que contém as respostas dos modelos e as respostas de referência (ground truth). "
        "Diversas métricas de similaridade (como Cosine Similarity, Difflib, BERTScore, entre outras) foram calculadas para cada comparação. "
        "O objetivo é entender a performance dos modelos a partir de diferentes perspectivas estatísticas e correlacionais."
    ))

# ---------------------------------------------------------------------------
# Seção 2. Metodologia
# ---------------------------------------------------------------------------
with doc.create(Section("Metodologia", numbering=False)):
    doc.append(NoEscape(
        "Inicialmente, os dados foram carregados e convertidos para um formato de dicionário. Foram construídos dois DataFrames principais: "
        "\\textbf{df\\_overall}: contém as estatísticas gerais (overall similarity) para cada questão e modelo; "
        "\\textbf{df\\_detailed}: armazena, para cada comparação (modelo versus ground truth), os valores de diversas métricas de similaridade. "
        "Em seguida, foram aplicadas análises descritivas, de correlação e variação, além da exportação dos dados para arquivos CSV e tabelas em LaTeX. "
        "Os gráficos gerados (barras, histogramas, boxplots, heatmaps e scatter plots) ilustram visualmente os resultados e auxiliam na interpretação dos dados."
    ))

# ---------------------------------------------------------------------------
# Seção 3. Resultados
# ---------------------------------------------------------------------------
with doc.create(Section("Resultados", numbering=False)):
    
    # 3.1 Estatísticas de Overall Similarity
    with doc.create(Subsection("Análise Descritiva da Similaridade Geral", numbering=False)):
        doc.append(NoEscape(
            "A tabela a seguir (Tabela \\ref{tab:overall_stats}) apresenta as estatísticas descritivas da overall similarity por modelo, "
            "incluindo média, mediana, desvio padrão, valor mínimo e máximo."
        ))
        doc.append(NoEscape(r"\begin{table}[H]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\caption{Estatísticas de Overall Similarity por Modelo}\label{tab:overall_stats}"))
        doc.append(NoEscape(overall_stats_table))
        doc.append(NoEscape(r"\end{table}"))
        
    # 3.2 Frequência de Scores Vazios
    with doc.create(Subsection("Frequência de Casos com Scores Vazios", numbering=False)):
        doc.append(NoEscape(
            "A tabela a seguir (Tabela \\ref{tab:empty_scores}) mostra a contagem e o percentual de casos com dicionários de scores vazios para cada modelo."
        ))
        doc.append(NoEscape(r"\begin{table}[H]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\caption{Frequência de Casos com Scores Vazios por Modelo}\label{tab:empty_scores}"))
        doc.append(NoEscape(empty_scores_table))
        doc.append(NoEscape(r"\end{table}"))
        
    # 3.3 Estatísticas Agregadas das Métricas Detalhadas
    with doc.create(Subsection("Estatísticas Agregadas das Métricas Detalhadas", numbering=False)):
        doc.append(NoEscape(
            "A tabela a seguir (Tabela \\ref{tab:aggregated_metrics}) apresenta as estatísticas agregadas das métricas detalhadas (média, mediana, desvio padrão, mínimo e máximo) para cada modelo."
        ))
        doc.append(NoEscape(r"\begin{table}[H]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\caption{Estatísticas Agregadas das Métricas Detalhadas por Modelo}\label{tab:aggregated_metrics}"))
        doc.append(NoEscape(aggregated_metrics_table))
        doc.append(NoEscape(r"\end{table}"))

# ---------------------------------------------------------------------------
# Seção 4. Discussão (texto gerado dinamicamente a partir dos dados)
# ---------------------------------------------------------------------------
with doc.create(Section("Discussão", numbering=False)):
    doc.append(discussion_text)

# ---------------------------------------------------------------------------
# Seção 5. Conclusões (texto gerado dinamicamente a partir dos dados)
# ---------------------------------------------------------------------------
with doc.create(Section("Conclusões", numbering=False)):
    doc.append(conclusion_text)

# ---------------------------------------------------------------------------
# Seção 6. Anexos
# ---------------------------------------------------------------------------
with doc.create(Section("Anexos", numbering=False)):
    doc.append(NoEscape(
        "Os anexos deste relatório incluem as tabelas geradas a partir dos arquivos CSV: "
        "\\texttt{overall\\_stats.csv}, \\texttt{empty\\_scores\\_frequency.csv}, e \\texttt{aggregated\\_metrics.csv}, "
        "além de todas as imagens geradas (gráficos em formato PNG presentes na pasta \\texttt{analysis\\_results})."
    ))

# ---------------------------------------------------------------------------
# Seção 7. Referências
# ---------------------------------------------------------------------------
with doc.create(Section("Referências", numbering=False)):
    doc.append(NoEscape(
        "Bibliotecas utilizadas: "
        "\\begin{itemize}"
        "\\item \\textbf{pandas}, \\textbf{numpy}: para manipulação e análise dos dados."
        "\\item \\textbf{matplotlib}, \\textbf{seaborn}: para geração dos gráficos."
        "\\item \\textbf{PyLaTeX}: para a criação deste relatório em LaTeX."
        "\\end{itemize} Além disso, é necessário ter instalada uma distribuição LaTeX (como TeX Live ou MiKTeX) para compilar o documento."
    ))

# Gerar o arquivo .tex
doc.generate_tex("RelatorioAnalise")

print("Arquivo RelatorioAnalise.tex gerado com sucesso!")
