import os
import pandas as pd
from pylatex import Document, Section, Subsection, Command, Figure, NoEscape, Package, MiniPage

# =============================================================================
# 1. Carregar as tabelas geradas pelas análises
# =============================================================================
csv_folder = "analysis_results"

# Carregar os CSVs com as estatísticas e métricas
overall_stats_df = pd.read_csv(os.path.join(csv_folder, "overall_stats.csv"), index_col=0)
empty_scores_df = pd.read_csv(os.path.join(csv_folder, "empty_scores_frequency.csv"), index_col=0)
aggregated_metrics_df = pd.read_csv(os.path.join(csv_folder, "aggregated_metrics.csv"), index_col=0)
variation_data_df = pd.read_csv(os.path.join(csv_folder, "variation_data.csv"), index_col=0)

# Função para escapar underscores (pois serão usados em LaTeX)
def escape_df(df):
    df.index = df.index.to_series().str.replace('_', r'\_', regex=False)
    df.columns = df.columns.str.replace('_', r'\_', regex=False)
    return df

overall_stats_df = escape_df(overall_stats_df)
empty_scores_df = escape_df(empty_scores_df)
aggregated_metrics_df = escape_df(aggregated_metrics_df)
variation_data_df = escape_df(variation_data_df)

# Converter DataFrames para tabelas LaTeX
overall_stats_table = overall_stats_df.to_latex(escape=False)
empty_scores_table = empty_scores_df.to_latex(escape=False)
aggregated_metrics_df = aggregated_metrics_df.round(4)
aggregated_metrics_df = aggregated_metrics_df.round(4)
aggregated_metrics_table = aggregated_metrics_df.to_latex(escape=False)
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
    
    
    
    # Análise de Casos "Unanswerable" com imagens lado a lado usando minipages
    with doc.create(Subsection("Análise de Casos 'Unanswerable'", numbering=False)):
        doc.append("Nesta parte, são analisados os casos em que os modelos não forneceram respostas (unanswerable). "
                "Os gráficos abaixo, apresentados lado a lado, mostram respectivamente a contagem desses casos e a relação entre a overall similarity e a condição de unanswerable. "
                "Em geral, uma baixa overall similarity pode estar associada a questões sem resposta.")
        with doc.create(Figure(position='H')) as unans_fig:
            unans_fig.append(NoEscape(r"\begin{minipage}[b]{0.45\textwidth}"))
            unans_fig.append(NoEscape(r"\centering"))
            unans_fig.append(NoEscape(r"\includegraphics[width=\linewidth]{" + str(os.path.join(csv_folder, "count_unanswerable.png")).replace(r'\\','/') + "}"))
            unans_fig.append(NoEscape(r"\caption*{Contagem de Casos Unanswerable.}"))
            unans_fig.append(NoEscape(r"\end{minipage}\hfill"))
            unans_fig.append(NoEscape(r"\begin{minipage}[b]{0.45\textwidth}"))
            unans_fig.append(NoEscape(r"\centering"))
            unans_fig.append(NoEscape(r"\includegraphics[width=\linewidth]{" + str(os.path.join(csv_folder, "boxplot_overall_vs_unanswerable.png")).replace(r'\\','/') + "}"))
            unans_fig.append(NoEscape(r"\caption*{Boxplot – Overall Similarity vs. Unanswerable.}"))
            unans_fig.append(NoEscape(r"\end{minipage}"))

    
    # Correlação Inter-Modelos
    with doc.create(Subsection("Correlação Inter-Modelos", numbering=False)):
        doc.append("Esta análise verifica a correlação da overall similarity entre os diferentes modelos. "
                   "Utilizando uma pivot table, foi calculada a correlação entre os modelos, cuja visualização por meio de um heatmap facilita a identificação de similaridades ou discrepâncias na performance entre eles.")
        with doc.create(Figure(position='H')) as heatmap_inter_fig:
            heatmap_inter_fig.add_image(os.path.join(csv_folder, 'heatmap_inter_model_corr.png'), width=NoEscape(r'0.8\textwidth'))
            heatmap_inter_fig.add_caption("Heatmap – Correlação Inter-Modelos de Overall Similarity.")
    
    # Estatísticas Agregadas das Métricas Detalhadas
    with doc.create(Subsection("Estatísticas Agregadas das Métricas Detalhadas", numbering=False)):
        doc.append("Por fim, a tabela a seguir apresenta as estatísticas agregadas (média, mediana, desvio padrão, mínimo e máximo) para cada métrica detalhada, "
                "agrupadas por modelo. Essa análise auxilia na identificação de padrões e na avaliação da consistência dos scores.")
        doc.append(NoEscape(r"\begin{table}[H]"))
        doc.append(NoEscape(r"\centering"))
        doc.append(NoEscape(r"\caption{Estatísticas Agregadas das Métricas Detalhadas por Modelo}\label{tab:aggregated_metrics}"))
        doc.append(NoEscape(aggregated_metrics_table))
        doc.append(NoEscape(r"\end{table}"))


# Discussão
with doc.create(Section("Discussão", numbering=False)):
    doc.append("Os resultados apresentados oferecem múltiplas perspectivas sobre a performance dos modelos de resposta. "
               "A análise das estatísticas de similaridade geral permite identificar quais modelos se destacam na aderência às respostas de referência, "
               "enquanto os gráficos de distribuição e boxplots revelam a variabilidade dos scores. "
               "A análise de correlação entre as métricas detalhadas e a overall similarity sugere a consistência (ou a falta dela) entre as diferentes formas de avaliação. "
               "Adicionalmente, a análise dos casos unanswerable e a variação entre ground truths evidenciam pontos que podem ser aprimorados nos algoritmos de resposta.")

# Conclusões
with doc.create(Section("Conclusões", numbering=False)):
    doc.append("Em resumo, o relatório demonstra que modelos com maiores médias de similaridade tendem a apresentar desempenho mais consistente. "
               "Entretanto, a presença de alta variabilidade e de casos unanswerable indica que há espaço para ajustes nos processos de geração e avaliação das respostas. "
               "A integração de diversas métricas e a análise visual proporcionam uma visão abrangente, que pode orientar futuras melhorias nos modelos.")

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
