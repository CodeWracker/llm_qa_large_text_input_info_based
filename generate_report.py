import os
import pandas as pd
from pathlib import Path
from pylatex import Document, Section, Subsection, Command, Figure, NoEscape, Package, MiniPage

# =============================================================================
# 1. Carregar as tabelas geradas pelas análises
# =============================================================================
csv_folder = "analysis_results"

# Carregar os CSVs com as estatísticas e métricas
overall_stats_df = pd.read_csv(os.path.join(csv_folder, "overall_stats.csv"), index_col=0)
empty_scores_df = pd.read_csv(os.path.join(csv_folder, "empty_scores_frequency.csv"), index_col=0)

variation_data_df = pd.read_csv(os.path.join(csv_folder, "variation_data.csv"), index_col=0)

# Função para escapar underscores (pois serão usados em LaTeX)
def escape_df(df):
    df.index = df.index.to_series().str.replace('_', r'\_', regex=False)
    df.columns = df.columns.str.replace('_', r'\_', regex=False)
    return df

overall_stats_df = escape_df(overall_stats_df)
empty_scores_df = escape_df(empty_scores_df)
variation_data_df = escape_df(variation_data_df)

# Converter DataFrames para tabelas LaTeX
overall_stats_table = overall_stats_df.to_latex(escape=False)
empty_scores_table = empty_scores_df.to_latex(escape=False)

# Define o caminho da pasta onde os CSVs estão armazenados
scores_folder = os.path.join("analysis_results", "scores")

# Lista todos os arquivos CSV na pasta scores
csv_files = [f for f in os.listdir(scores_folder) if f.endswith('.csv')]

# Função para formatar cada valor para 4 casas decimais, se possível
def format_value(x):
    try:
        return f"{float(x):.4f}"
    except (ValueError, TypeError):
        return x

# Lista para armazenar as tabelas LaTeX de cada CSV
detailed_results_latex_tables = []

# Itera sobre cada arquivo CSV encontrado
for csv_file in csv_files:
    file_path = os.path.join(scores_folder, csv_file)
    df = pd.read_csv(file_path, index_col=0)
    # Aplica a formatação célula por célula
    df = df.applymap(format_value)
    # Converte o DataFrame para uma tabela em LaTeX
    latex_table = df.to_latex(escape=False)
    detailed_results_latex_tables.append(latex_table)

variation_data_table = variation_data_df.to_latex(escape=False)

# =============================================================================
# 2. Configurar o documento LaTeX
# =============================================================================
geometry_options = {
    "tmargin": "2cm",
    "lmargin": "2cm",
    "rmargin": "2cm",
    "bmargin": "2cm"
}
doc = Document("RelatorioAnalise", geometry_options=geometry_options)

# Adicionar pacotes necessários
doc.packages.append(Package("graphicx"))
doc.packages.append(Package("float"))
doc.packages.append(Package("booktabs"))
doc.packages.append(Package("rotating"))

# Pré-ambiente: título, autor, data
doc.preamble.append(Command("title", "Relatório de Análise de Performance dos Modelos de Resposta"))
doc.preamble.append(Command("date", NoEscape(r"\today")))
doc.append(NoEscape(r"\maketitle"))

# =============================================================================
# 3. Seções do relatório
# =============================================================================

# Introdução
with doc.create(Section("Introdução", numbering=False)):
    doc.append("Este relatório apresenta uma análise completa dos modelos de resposta, avaliando a similaridade entre as respostas geradas e as respostas de referência. "
               "Foram calculadas diversas métricas de similaridade e realizadas análises estatísticas e visuais para identificar padrões de performance, inconsistências e possíveis pontos de melhoria. "
               "O objetivo é fornecer subsídios para interpretar os dados de forma objetiva, auxiliando na tomada de decisões para ajustes nos algoritmos.")

# Metodologia
with doc.create(Section("Metodologia", numbering=False)):
    doc.append("A análise foi realizada em duas etapas principais:\n\n")
    doc.append(NoEscape(r"\begin{itemize}"))
    doc.append(NoEscape(r"\item Carregamento dos dados: os dados foram extraídos de um arquivo pickle e convertidos para dicionários, possibilitando o acesso às respostas dos modelos e às respostas de referência."))
    doc.append(NoEscape(r"\item Processamento e análise: foram calculadas diversas métricas de similaridade, estatísticas descritivas e correlações. Além disso, foram gerados diversos gráficos (barras, histogramas, boxplots, heatmaps e scatter plots) para facilitar a interpretação visual dos resultados."))
    doc.append(NoEscape(r"\end{itemize}"))
    doc.append("Cada gráfico ou tabela vem acompanhado de uma breve explicação sobre como interpretá-lo, de modo que o leitor possa compreender os pontos principais de cada análise.")

# Resultados
with doc.create(Section("Resultados", numbering=False)):
    
    # Estatísticas de Similaridade Geral
    with doc.create(Subsection("Estatísticas de Similaridade Geral", numbering=False)):
        doc.append("Nesta seção são apresentadas as estatísticas descritivas da overall similarity dos modelos. "
                   "Valores médios e medianos mais elevados indicam melhor aderência às respostas de referência, enquanto um alto desvio padrão pode indicar maior variabilidade.")
        
        doc.append("\nForam gerados os seguintes gráficos para visualizar esses dados:")
        with doc.create(Figure(position='H')) as bar_fig:
            bar_fig.add_image(os.path.join(csv_folder, 'barplot_overall_similarity.png'), width=NoEscape(r'0.8\textwidth'))
            bar_fig.add_caption("Gráfico de Barras – Média de Overall Similarity por Modelo. Observe as diferenças de desempenho entre os modelos.")
        with doc.create(Figure(position='H')) as hist_fig:
            hist_fig.add_image(os.path.join(csv_folder, 'histogram_overall_similarity.png'), width=NoEscape(r'0.8\textwidth'))
            hist_fig.add_caption("Histograma – Distribuição de Overall Similarity. Permite visualizar a dispersão dos valores obtidos.")
        with doc.create(Figure(position='H')) as box_fig:
            box_fig.add_image(os.path.join(csv_folder, 'boxplot_overall_similarity.png'), width=NoEscape(r'0.8\textwidth'))
            box_fig.add_caption("Boxplot – Overall Similarity por Modelo. Facilita a identificação de mediana, quartis e outliers.")
    
    # Correlação entre Métricas de Similaridade
    with doc.create(Subsection("Correlação entre Métricas de Similaridade", numbering=False)):
        doc.append("Nesta análise, avaliamos a correlação entre as diferentes métricas de similaridade (score keys). "
                   "O heatmap abaixo mostra a força da correlação entre cada par de métricas: valores próximos de 1 ou -1 indicam forte correlação, enquanto valores próximos de 0 sugerem correlação fraca ou inexistente.")
        with doc.create(Figure(position='H')) as heatmap_fig:
            heatmap_fig.add_image(os.path.join(csv_folder, 'heatmap_score_keys_correlation.png'), width=NoEscape(r'0.8\textwidth'))
            heatmap_fig.add_caption("Heatmap – Correlação entre as Métricas de Similaridade.")
        doc.append("\nAlém do heatmap, foram gerados scatter plots individuais comparando cada métrica com a overall similarity. Esses gráficos podem ser consultados nos anexos para análises mais detalhadas.")
    
    # Correlação entre Overall Similarity e Média dos Scores (avg_metric)
    with doc.create(Subsection("Correlação entre Overall Similarity e Média dos Scores", numbering=False)):
        doc.append("Esta seção apresenta a relação entre a overall similarity e a média dos scores detalhados (avg\_metric). "
                   "O scatter plot a seguir permite identificar se existe uma tendência linear entre essas duas medidas, o que indicaria consistência na avaliação dos modelos.")
        with doc.create(Figure(position='H')) as scatter_avg_fig:
            scatter_avg_fig.add_image(os.path.join(csv_folder, 'scatter_overall_vs_avg_metric.png'), width=NoEscape(r'0.8\textwidth'))
            scatter_avg_fig.add_caption("Scatter Plot – Overall Similarity vs. Média dos Score Metrics (avg\_metric).")
    
    
    
    # Converter os caminhos das imagens para o formato POSIX
    count_image_path = str(Path(csv_folder, "count_unanswerable.png").as_posix())
    box_image_path = str(Path(csv_folder, "boxplot_overall_vs_unanswerable.png").as_posix())

    # Análise de Casos "Unanswerable" com imagens lado a lado usando minipages
    with doc.create(Subsection("Análise de Casos 'Unanswerable'", numbering=False)):
        doc.append("Nesta parte, são analisados os casos em que os modelos não forneceram respostas (unanswerable). "
                "Os gráficos abaixo, apresentados lado a lado, mostram respectivamente a contagem desses casos e a relação entre a overall similarity e a condição de unanswerable. "
                "Em geral, uma baixa overall similarity pode estar associada a questões sem resposta.")
        with doc.create(Figure(position='H')) as unans_fig:
            unans_fig.append(NoEscape(r"\begin{minipage}[b]{0.45\textwidth}"))
            unans_fig.append(NoEscape(r"\centering"))
            unans_fig.append(NoEscape(r"\includegraphics[width=\linewidth]{" + count_image_path + "}"))
            unans_fig.append(NoEscape(r"\caption*{Contagem de Casos Unanswerable.}"))
            unans_fig.append(NoEscape(r"\end{minipage}\hfill"))
            unans_fig.append(NoEscape(r"\begin{minipage}[b]{0.45\textwidth}"))
            unans_fig.append(NoEscape(r"\centering"))
            unans_fig.append(NoEscape(r"\includegraphics[width=\linewidth]{" + box_image_path + "}"))
            unans_fig.append(NoEscape(r"\caption*{Boxplot – Overall Similarity vs. Unanswerable.}"))
            unans_fig.append(NoEscape(r"\end{minipage}"))

    
    # Correlação Inter-Modelos
    with doc.create(Subsection("Correlação Inter-Modelos", numbering=False)):
        doc.append("Esta análise verifica a correlação da overall similarity entre os diferentes modelos. "
                   "Utilizando uma pivot table, foi calculada a correlação entre os modelos, cuja visualização por meio de um heatmap facilita a identificação de similaridades ou discrepâncias na performance entre eles.")
        with doc.create(Figure(position='H')) as heatmap_inter_fig:
            heatmap_inter_fig.add_image(os.path.join(csv_folder, 'heatmap_inter_model_corr.png'), width=NoEscape(r'0.8\textwidth'))
            heatmap_inter_fig.add_caption("Heatmap – Correlação Inter-Modelos de Overall Similarity.")
    
# Seção: Estatísticas Agregadas das Métricas Detalhadas
with doc.create(Subsection("Estatísticas Agregadas das Métricas Detalhadas", numbering=False)):
    doc.append("A seguir, serão apresentadas diversas tabelas, cada uma correspondendo a um score específico. "
               "Em cada tabela, estão exibidas as estatísticas agregadas – média, mediana, desvio padrão, valor mínimo e valor máximo – "
               "agrupadas por modelo. Essas métricas auxiliam na identificação de padrões e na avaliação da consistência dos scores obtidos.")
    
    # Itera sobre as tabelas geradas (uma para cada score)
    for i, table in enumerate(detailed_results_latex_tables):
        # Usa o nome do arquivo CSV para identificar o score
        metric_name = csv_files[i].replace(".csv", "").replace("_", r"\_")
        doc.append(NoEscape(r"\subsubsection*{Estatísticas para o Score: " + metric_name + "}"))
        doc.append(NoEscape(r"\begin{table}[H]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(table))
        doc.append(NoEscape(r"\end{table}"))
        doc.append(NoEscape(r"\vspace{0.5cm}"))


# ============================================================================
# Gerar análises simples para as seções de Discussão 
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


# Discussão (texto gerado dinamicamente a partir dos dados)
# ---------------------------------------------------------------------------
with doc.create(Section("Discussão", numbering=False)):
    doc.append(discussion_text)


# Anexos
with doc.create(Section("Anexos", numbering=False)):
    doc.append("Os anexos deste relatório reúnem os arquivos CSV gerados e todos os gráficos produzidos durante a análise, tais como:")
    doc.append(NoEscape(r"\begin{itemize}"))
    doc.append(NoEscape(r"\item overall\_stats.csv"))
    doc.append(NoEscape(r"\item empty\_scores\_frequency.csv"))
    doc.append(NoEscape(r"\item variation\_data.csv"))
    doc.append(NoEscape(r"\item aggregated\_metrics.csv"))
    doc.append(NoEscape(r"\item overall\_similarity.csv"))
    doc.append(NoEscape(r"\item detailed\_similarity.csv"))
    doc.append(NoEscape(r"\end{itemize}"))
    doc.append("Além disso, todos os gráficos (barplots, histogramas, boxplots, heatmaps e scatter plots) podem ser conferidos na pasta 'analysis_results'.")

# Referências
with doc.create(Section("Referências", numbering=False)):
    doc.append("As principais bibliotecas e ferramentas utilizadas para a geração deste relatório foram:")
    doc.append(NoEscape(r"\begin{itemize}"))
    doc.append(NoEscape(r"\item \textbf{pandas} e \textbf{numpy} para manipulação e análise dos dados."))
    doc.append(NoEscape(r"\item \textbf{matplotlib} e \textbf{seaborn} para a criação dos gráficos."))
    doc.append(NoEscape(r"\item \textbf{PyLaTeX} para a montagem do documento LaTeX."))
    doc.append(NoEscape(r"\end{itemize}"))

# =============================================================================
# 4. Gerar o arquivo .tex
# =============================================================================
doc.generate_tex("RelatorioAnalise")

print("Arquivo RelatorioAnalise.tex gerado com sucesso!")
