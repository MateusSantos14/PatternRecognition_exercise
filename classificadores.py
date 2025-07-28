# Arquivo: classificadores.py
import numpy as np
import time
from collections import defaultdict

def quadratico(data, num_runs, train_percent):
    
    # --- 1. Inicialização e extração de dados ---
    n_total_samples, p_plus_1 = data.shape
    num_features = p_plus_1 - 1
    
    # Separa features (X) e rótulos (y) uma única vez
    X_data = data[:, :-1]
    y_data = data[:, -1]
    
    num_train = round(train_percent * n_total_samples / 100)
    num_test = n_total_samples - num_train
    
    # Encontra o número de classes a partir do dataset completo
    num_classes = int(np.max(y_data))
    print(f'O problema tem {num_classes} classes, {num_features} features.')
    
    accuracy_scores = np.zeros(num_runs)
    
    # Variáveis para armazenar os parâmetros da última execução
    means_last_run, covs_last_run, ranks_last_run = [], [], []

    # --- 2. Laço principal de execuções (Monte Carlo) ---
    for r in range(num_runs):
        
        # Embaralha os índices e divide os dados
        permutation = np.random.permutation(n_total_samples)
        train_indices = permutation[:num_train]
        test_indices = permutation[num_train:]
        
        X_train, y_train = X_data[train_indices], y_data[train_indices]
        X_test, y_test = X_data[test_indices], y_data[test_indices]
        
        # Dicionários para armazenar os parâmetros desta iteração
        means = {}
        inv_covs = {}
        log_dets = {}
        log_priors = {}
        
        unique_classes_in_train = np.unique(y_train)

        # --- 3. Fase de Treinamento ---
        for k in unique_classes_in_train:
            X_k = X_train[y_train == k]
            
            # Cálculo dos parâmetros da classe k
            m_k = np.mean(X_k, axis=0)
            S_k = np.cov(X_k, rowvar=False)
            iS_k = np.linalg.pinv(S_k) # Pseudo-inversa para estabilidade numérica
            sign, log_det_S_k = np.linalg.slogdet(S_k)
            
            if sign <= 0:
                log_det_S_k = -np.inf # Evita log de determinante não-positivo

            prior_k = X_k.shape[0] / num_train
            
            # Armazena os parâmetros nos dicionários
            means[k] = m_k
            inv_covs[k] = iS_k
            log_dets[k] = log_det_S_k
            log_priors[k] = np.log(prior_k)

            # Se for a última execução, armazena os parâmetros para retorno
            if r == num_runs - 1:
                means_last_run.append(m_k)
                covs_last_run.append(S_k)
                ranks_last_run.append(np.linalg.matrix_rank(S_k))

        # --- 4. Fase de Teste ---
        discriminant_scores = np.zeros((num_test, len(unique_classes_in_train)))
        
        # Itera sobre as classes encontradas no treino para calcular os scores
        for i, k in enumerate(unique_classes_in_train):
            diff = X_test - means[k]
            
            # Distância de Mahalanobis (vetorizada)
            mahalanobis_dist = np.sum((diff @ inv_covs[k]) * diff, axis=1)
            
            # Score final do discriminante
            discriminant_scores[:, i] = -0.5 * mahalanobis_dist - 0.5 * log_dets[k] + log_priors[k]
        
        # A predição é a classe com o maior score
        predicted_class_indices = np.argmax(discriminant_scores, axis=1)
        predicted_labels = unique_classes_in_train[predicted_class_indices]
        
        # --- 5. Cálculo da Acurácia ---
        correct = np.sum(predicted_labels == y_test)
        accuracy_scores[r] = 100 * correct / num_test

    # --- 6. Consolidação dos Resultados ---
    stats = np.array([
        np.mean(accuracy_scores), 
        np.min(accuracy_scores), 
        np.max(accuracy_scores), 
        np.median(accuracy_scores), 
        np.std(accuracy_scores)
    ])
    
    print("\n--- Estatísticas Finais da Acurácia (%) ---")
    print(f"Média  : {stats[0]:.2f}")
    print(f"Mínimo : {stats[1]:.2f}")
    print(f"Máximo : {stats[2]:.2f}")
    print(f"Mediana: {stats[3]:.2f}")
    print(f"Desvio Padrão: {stats[4]:.2f}")

    return stats, accuracy_scores, means_last_run, covs_last_run, ranks_last_run




# --- Placeholders para os outros classificadores ---

def variante1(data, num_runs, train_percent, reg_param):
    
    # --- 1. Inicialização e extração de dados ---
    n_total_samples, p_plus_1 = data.shape
    num_features = p_plus_1 - 1
    
    # Separa features (X) e rótulos (y) uma única vez
    X_data = data[:, :-1]
    y_data = data[:, -1]
    
    num_train = round(train_percent * n_total_samples / 100)
    num_test = n_total_samples - num_train
    
    # Encontra o número de classes a partir do dataset completo
    num_classes = int(np.max(y_data))
    print(f'O problema tem {num_classes} classes, {num_features} features.')
    
    accuracy_scores = np.zeros(num_runs)
    
    # Variáveis para armazenar os parâmetros da última execução
    means_last_run, covs_last_run, ranks_last_run = [], [], []

    # --- 2. Laço principal de execuções (Monte Carlo) ---
    for r in range(num_runs):
        
        # Embaralha os índices e divide os dados
        permutation = np.random.permutation(n_total_samples)
        train_indices = permutation[:num_train]
        test_indices = permutation[num_train:]
        
        X_train, y_train = X_data[train_indices], y_data[train_indices]
        X_test, y_test = X_data[test_indices], y_data[test_indices]
        
        # Dicionários para armazenar os parâmetros desta iteração
        means = {}
        inv_covs = {}
        log_dets = {}
        log_priors = {}
        
        unique_classes_in_train = np.unique(y_train)

        # --- 3. Fase de Treinamento ---
        for k in unique_classes_in_train:
            X_k = X_train[y_train == k]
            
            # Cálculo dos parâmetros da classe k
            m_k = np.mean(X_k, axis=0)
            S_k = np.cov(X_k, rowvar=False)
            iS_k = np.linalg.pinv(S_k) # Pseudo-inversa para estabilidade numérica
            sign, log_det_S_k = np.linalg.slogdet(S_k)
            
            if sign <= 0:
                log_det_S_k = -np.inf # Evita log de determinante não-positivo

            prior_k = X_k.shape[0] / num_train
            
            # Armazena os parâmetros nos dicionários
            means[k] = m_k
            inv_covs[k] = iS_k
            log_dets[k] = log_det_S_k
            log_priors[k] = np.log(prior_k)

            # Se for a última execução, armazena os parâmetros para retorno
            if r == num_runs - 1:
                means_last_run.append(m_k)
                covs_last_run.append(S_k)
                ranks_last_run.append(np.linalg.matrix_rank(S_k))

        # --- 4. Fase de Teste ---
        discriminant_scores = np.zeros((num_test, len(unique_classes_in_train)))
        
        # Itera sobre as classes encontradas no treino para calcular os scores
        for i, k in enumerate(unique_classes_in_train):
            diff = X_test - means[k]
            
            # Distância de Mahalanobis (vetorizada)
            mahalanobis_dist = np.sum((diff @ inv_covs[k]) * diff, axis=1)
            
            # Score final do discriminante
            discriminant_scores[:, i] = -0.5 * mahalanobis_dist - 0.5 * log_dets[k] + log_priors[k]
        
        # A predição é a classe com o maior score
        predicted_class_indices = np.argmax(discriminant_scores, axis=1)
        predicted_labels = unique_classes_in_train[predicted_class_indices]
        
        # --- 5. Cálculo da Acurácia ---
        correct = np.sum(predicted_labels == y_test)
        accuracy_scores[r] = 100 * correct / num_test

    # --- 6. Consolidação dos Resultados ---
    stats = np.array([
        np.mean(accuracy_scores), 
        np.min(accuracy_scores), 
        np.max(accuracy_scores), 
        np.median(accuracy_scores), 
        np.std(accuracy_scores)
    ])
    
    print("\n--- Estatísticas Finais da Acurácia (%) ---")
    print(f"Média  : {stats[0]:.2f}")
    print(f"Mínimo : {stats[1]:.2f}")
    print(f"Máximo : {stats[2]:.2f}")
    print(f"Mediana: {stats[3]:.2f}")
    print(f"Desvio Padrão: {stats[4]:.2f}")

    return stats, accuracy_scores, means_last_run, covs_last_run, ranks_last_run



def variante2(D, Nr, Ptrain):
    print("Executando: Variante 2 (placeholder)")
    stats = np.random.rand(5) * 100
    tx_ok = np.random.normal(94, 3.0, Nr)
    return stats, tx_ok, None, None, None, None

def variante3(D, Nr, Ptrain, reg_param):
    print(f"Executando: Variante 3 com regularização={reg_param} (placeholder)")
    stats = np.random.rand(5) * 100
    tx_ok = np.random.normal(95.5, 2.2, Nr)
    return stats, tx_ok, None, None, None, None

def variante4(D, Nr, Ptrain):
    print("Executando: Variante 4 - Naive Bayes (placeholder)")
    stats = np.random.rand(5) * 100
    tx_ok = np.random.normal(90, 4.0, Nr)
    return stats, tx_ok, None, None, None, None

def linearMQ(D, Nr, Ptrain):
    print("Executando: Mínimos Quadrados (placeholder)")
    stats = np.random.rand(5) * 100
    tx_ok = np.random.normal(88, 5.0, Nr)
    return stats, tx_ok, None