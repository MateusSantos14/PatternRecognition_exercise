import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseClassifier(ABC):
    """
    Classe base abstrata para classificadores.
    
    Implementa um harness de avaliação completo, incluindo a divisão de dados,
    múltiplas execuções (rodadas) e cálculo de estatísticas de desempenho.
    Classes filhas precisam apenas implementar os métodos _train e _predict.
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None
        # Atributos específicos do modelo podem ser definidos em _train
        self.model_params = {}

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Treina o classificador. Deve ser implementado pela classe filha.
        
        Parâmetros:
        - X_train: Amostras de treino (features).
        - y_train: Rótulos de treino.
        """
        pass

    @abstractmethod
    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prevê os rótulos para as amostras de teste. Deve ser implementado pela classe filha.

        Parâmetros:
        - X_test: Amostras de teste (features).

        Retorna:
        - Array com os rótulos previstos.
        """
        pass

    def evaluate(self, data: np.ndarray, num_runs: int, train_percent: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executa a avaliação completa do classificador.

        Parâmetros:
        - data: Array numpy completo com features e o rótulo na última coluna.
        - num_runs: Número de execuções (rodadas) para a avaliação.
        - train_percent: Percentual de dados a serem usados para treinamento.

        Retorna:
        - STATS: Estatísticas de acurácia (média, min, max, mediana, desvio padrão).
        - TX_OK: Vetor com a taxa de acerto de cada uma das execuções.
        - last_run_info: Dicionário contendo informações da última execução (modelo, dados, etc).
        """
        n_total_samples, p_plus_1 = data.shape
        X_data = data[:, :-1]
        y_data = data[:, -1].astype(int)

        n_train = round(train_percent * n_total_samples / 100)
        n_test = n_total_samples - n_train
        
        accuracy_rates = np.zeros(num_runs)
        
        print(f"Executando {self.__class__.__name__}...")

        for r in range(num_runs):
            permutation = np.random.permutation(n_total_samples)
            train_indices = permutation[:n_train]
            test_indices = permutation[n_train:]

            self.X_train, self.y_train = X_data[train_indices], y_data[train_indices]
            X_test, y_test = X_data[test_indices], y_data[test_indices]
            
            # Garante que todas as classes estão presentes no treino para evitar erros
            if len(np.unique(self.y_train)) != len(np.unique(y_data)):
                # Em um caso real, poderíamos forçar a estratificação ou pular a rodada
                continue

            self._train(self.X_train, self.y_train)
            predicted_labels = self._predict(X_test)

            correct = np.sum(predicted_labels == y_test)
            accuracy_rates[r] = 100 * correct / n_test

        stats = np.array([
            np.mean(accuracy_rates), np.min(accuracy_rates), np.max(accuracy_rates),
            np.median(accuracy_rates), np.std(accuracy_rates)
        ])
        
        last_run_info = {
            "model_params": self.model_params,
            "X_train": self.X_train,
            "y_train": self.y_train
        }

        print(f"Finalizado: {self.__class__.__name__}. Acurácia Média: {stats[0]:.2f}%")
        return stats, accuracy_rates, last_run_info

# --- 1. Distância Mínima do Centroide (DMC) ---
class DMC(BaseClassifier):
    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model_params['centroids'] = {}
        self.model_params['class_labels'] = np.unique(y_train)
        for k in self.model_params['class_labels']:
            self.model_params['centroids'][k] = np.mean(X_train[y_train == k], axis=0)

    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        centroids = self.model_params['centroids']
        class_labels = self.model_params['class_labels']
        distances = np.zeros((X_test.shape[0], len(class_labels)))
        
        for i, k in enumerate(class_labels):
            distances[:, i] = np.linalg.norm(X_test - centroids[k], axis=1)
            
        predicted_indices = np.argmin(distances, axis=1)
        return class_labels[predicted_indices]

# --- 2. 1-Vizinho Mais Próximo (1-NN) - Otimizado ---
class NN1(BaseClassifier):
    """
    Implementação do 1-Vizinho Mais Próximo (1-NN) com busca de força bruta
    otimizada por vetorização NumPy.
    """
    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model_params['X_train'] = X_train
        self.model_params['y_train'] = y_train

    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        X_train = self.model_params['X_train']
        y_train = self.model_params['y_train']

        y_pred = np.empty(X_test.shape[0], dtype=y_train.dtype)


        for i, x_test_point in enumerate(X_test):
            diferencas = X_train - x_test_point

            diferencas_quadradas = diferencas ** 2

            distancias_quadradas = np.sum(diferencas_quadradas, axis=1)

            indice_vizinho_proximo = np.argmin(distancias_quadradas)
            
            y_pred[i] = y_train[indice_vizinho_proximo]

        return y_pred

# --- 3. Máxima Correlação (MaxCorr) ---
class MaxCorr(BaseClassifier):
    def _train(self, X_train, y_train):
        self.model_params['norm_centroids'] = {}
        self.model_params['class_labels'] = np.unique(y_train)
        for k in self.model_params['class_labels']:
            centroid = np.mean(X_train[y_train == k], axis=0)
            norm = np.linalg.norm(centroid)
            self.model_params['norm_centroids'][k] = centroid / (norm + 1e-9)

    def _predict(self, X_test):
        norm_centroids = self.model_params['norm_centroids']
        class_labels = self.model_params['class_labels']
        
        norms_X_test = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_test_norm = X_test / (norms_X_test + 1e-9)
        
        correlations = np.zeros((X_test.shape[0], len(class_labels)))
        
        for i, k in enumerate(class_labels):
            correlations[:, i] = X_test_norm @ norm_centroids[k]
        
        predicted_indices = np.argmax(correlations, axis=1)
        return class_labels[predicted_indices]

# --- 4. Classificador Quadrático (QDA) e suas Variantes ---
class QDA(BaseClassifier):
    def __init__(self, regularization_param=None, mode='QDA'):
        super().__init__()
        self.regularization_param = regularization_param
        self.mode = mode # 'QDA', 'Tikhonov', 'LDA', 'Friedman', 'NaiveBayes'
        
    def _train(self, X_train, y_train):
        num_features = X_train.shape[1]
        self.model_params['class_labels'] = np.unique(y_train)
        
        # --- MODIFICAÇÃO INICIAL ---
        # Garante que o dicionário de parâmetros do modelo conterá as matrizes de covariância.
        self.model_params['covariances'] = {}
        # ---------------------------

        self.model_params['means'] = {}
        self.model_params['inv_covs'] = {}
        self.model_params['log_dets'] = {}
        self.model_params['log_priors'] = {}

        # Lógica para variantes que precisam de uma matriz de covariância pooled
        pooled_cov = np.zeros((num_features, num_features))
        # Armazenamento temporário das covariâncias originais para o modo Friedman
        friedman_raw_covs = {}

        for k in self.model_params['class_labels']:
            X_k = X_train[y_train == k]
            n_k = X_k.shape[0]
            prior = n_k / len(y_train)
            
            self.model_params['means'][k] = np.mean(X_k, axis=0)
            self.model_params['log_priors'][k] = np.log(prior)
            
            cov_k = np.cov(X_k, rowvar=False)

            # Seleciona e modifica a matriz de covariância com base no modo
            if self.mode == 'Tikhonov':
                cov_k += np.eye(num_features) * self.regularization_param
            elif self.mode in ['LDA', 'Friedman']:
                pooled_cov += cov_k * prior
                if self.mode == 'Friedman':
                    friedman_raw_covs[k] = cov_k # Salva a matriz original para cálculo posterior
            elif self.mode == 'NaiveBayes':
                cov_k = np.diag(np.diag(cov_k))
            
            self.model_params['covariances'][k] = cov_k

            sign, log_det = np.linalg.slogdet(cov_k)
            self.model_params['inv_covs'][k] = np.linalg.pinv(cov_k)
            self.model_params['log_dets'][k] = log_det

        if self.mode == 'LDA':
            sign, log_det = np.linalg.slogdet(pooled_cov)
            for k in self.model_params['class_labels']:
                # --- MODIFICAÇÃO (PARTE 2) ---
                # Salva a matriz de covariância pooled para cada classe no modo LDA.
                self.model_params['covariances'][k] = pooled_cov
                # ---------------------------
                self.model_params['inv_covs'][k] = np.linalg.pinv(pooled_cov)
                self.model_params['log_dets'][k] =  0 # Log det se cancela na função discriminante

        if self.mode == 'Friedman':
            for k in self.model_params['class_labels']:
                X_k = X_train[y_train == k]
                n_k = X_k.shape[0]
                c = friedman_raw_covs[k]
                lamba_parameter = self.regularization_param
                cov_k_numerator = (1-lamba_parameter)*(n_k*c) + lamba_parameter * (pooled_cov*len(X_train))
                cov_k_denominator = (1 - lamba_parameter)*n_k + lamba_parameter * len(X_train)

                cov_k = cov_k_numerator/cov_k_denominator
                
                # --- MODIFICAÇÃO (PARTE 3) ---
                # Salva a matriz de covariância regularizada final para o modo Friedman.
                self.model_params['covariances'][k] = cov_k
                # ---------------------------

                sign, log_det = np.linalg.slogdet(cov_k)
                self.model_params['inv_covs'][k] = np.linalg.pinv(cov_k)
                self.model_params['log_dets'][k] = np.linalg.slogdet(cov_k)[1]

    def _predict(self, X_test):
        class_labels = self.model_params['class_labels']
        scores = np.zeros((X_test.shape[0], len(class_labels)))

        for i, k in enumerate(class_labels):
            diff = X_test - self.model_params['means'][k]
            mahalanobis = np.sum((diff @ self.model_params['inv_covs'][k]) * diff, axis=1)
            scores[:, i] = mahalanobis + self.model_params['log_dets'][k] - 2 * self.model_params['log_priors'][k]

        predicted_indices = np.argmin(scores, axis=1)
        return class_labels[predicted_indices]
