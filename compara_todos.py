import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython.display import display
from models import DMC,NN1,MaxCorr,QDA
def compara_todos(nome_arquivo: str, Nr: int, Ptrain: float):
    """
    Executa uma análise comparativa de múltiplos classificadores de aprendizado de máquina.

    A função carrega um conjunto de dados, treina e avalia diversos classificadores,
    mede o tempo de execução, e apresenta os resultados em uma tabela e um boxplot.

    Args:
        nome_arquivo (str): O caminho para o arquivo de dados (ex: 'recfaces400.dat').
        Nr (int): O número de repetições (ou "realizações") do experimento para cada classificador.
        Ptrain (float): A proporção dos dados a ser usada para treinamento (ex: 0.8 para 80%).

    Returns:
        pandas.DataFrame: Um DataFrame contendo as estatísticas de desempenho e tempo de
                          execução para cada classificador. Retorna None se o arquivo
                          de dados não for encontrado.
    """
    # --- Carregamento dos dados ---
    try:
        print(f"--- CARREGANDO DADOS DE '{nome_arquivo}' ---")
        D = np.loadtxt(nome_arquivo)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo '{nome_arquivo}' não encontrado.")
        print("A execução foi interrompida.")
        return None

    # --- Definição dos Classificadores ---
    # A ordem aqui define a ordem na tabela e no gráfico
    classifiers = {
        "QDA (Quadrático)": QDA(mode='QDA'),
        "Variante 1 (Tikhonov λ=0.01)": QDA(mode='Tikhonov', regularization_param=0.01),
        "Variante 2 (Covariance Pooled)": QDA(mode='LDA'),
        "Variante 3 (Friedman λ=0.5)": QDA(mode='Friedman', regularization_param=0.5),
        "Variante 4 (Naive Bayes)": QDA(mode='NaiveBayes'),
        "MaxCorr": MaxCorr(),
        "DMC": DMC(),
        "1-NN": NN1()
    }

    # --- Execução, Medição de Tempo e Coleta de Resultados ---
    all_stats = []
    all_tx_ok = []
    all_times = []
    all_last_run_info = {}
    classifier_names = list(classifiers.keys())

    print(f"\n--- INICIANDO AVALIAÇÃO DOS CLASSIFICADORES ({Nr} repetições, {Ptrain}% treino) ---")
    for name, classifier in classifiers.items():
        print(f"Avaliando: {name}...")
        start_time = time.time()
        # A função evaluate recebe os parâmetros Nr e Ptrain
        print(f"Classificador: {name}")
        stats, tx_ok, last_run_info = classifier.evaluate(D, Nr, Ptrain)
        all_last_run_info[name] = last_run_info
        execution_time = time.time() - start_time
        
        all_stats.append(stats)
        all_tx_ok.append(tx_ok)
        all_times.append(execution_time)
    print("\n--- AVALIAÇÃO FINALIZADA ---\n")

    # --- Montagem e Exibição do DataFrame de Resultados ---
    stats_array = np.array(all_stats)
    data_for_df = {
        'Média (%)': stats_array[:, 0],
        'Mínimo (%)': stats_array[:, 1],
        'Máximo (%)': stats_array[:, 2],
        'Mediana (%)': stats_array[:, 3],
        'Desvio Padrão': stats_array[:, 4],
        'Tempo (s)': all_times
    }

    df_results = pd.DataFrame(data_for_df, index=classifier_names)
    pd.options.display.float_format = '{:,.2f}'.format

    print("--- TABELA DE RESULTADOS ---")
    display(df_results)

    # --- Plotagem do Boxplot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    box = plt.boxplot(all_tx_ok, patch_artist=True, labels=classifier_names)

    # Adiciona cores para melhor visualização
    colors = plt.cm.get_cmap('viridis', len(all_tx_ok))
    for patch, color in zip(box['boxes'], colors(np.linspace(0, 1, len(all_tx_ok)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.title(f'Comparação de Desempenho dos Classificadores (Dataset: {nome_arquivo})', fontsize=16, pad=20)
    plt.ylabel('Taxa de Acerto (%)', fontsize=12)
    plt.xticks(rotation=45, ha="right") # Rotaciona os labels para não sobrepor
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return (df_results,all_last_run_info)