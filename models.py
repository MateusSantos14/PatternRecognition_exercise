import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseClassifier(ABC):
    """Classe base abstrata para classificadores.

    Esta classe fornece um sistema de avaliação completo, incluindo a divisão de dados,
    múltiplas execuções (rodadas) e o cálculo de estatísticas de desempenho.
    Classes filhas precisam apenas implementar os métodos `_train` e `_predict`.
    """
    def __init__(self):
        """Inicializa o classificador."""
        self.X_train = None
        self.y_train = None
        self.model_params = {}

    @abstractmethod
    def _train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o classificador usando os dados de treino fornecidos.

        Parâmetros:
            X_train: As amostras de treino (features).
            y_train: Os rótulos de treino.
        """
        pass

    @abstractmethod
    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prevê os rótulos de classe para as amostras de teste fornecidas.

        Parâmetros:
            X_test: As amostras de teste (features).

        Retorna:
            Um array contendo os rótulos previstos para cada amostra de teste.
        """
        pass

    def evaluate(self, data: np.ndarray, num_runs: int, train_percent: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Executa uma avaliação completa do classificador.

        Este método divide os dados em conjuntos de treino e teste, executa o
        processo de treino e predição múltiplas vezes e calcula as métricas de desempenho.

        Parâmetros:
            data: O conjunto de dados completo, com as features nas colunas
                  iniciais e o rótulo da classe na última coluna.
            num_runs: O número de vezes para repetir a avaliação (rodadas).
            train_percent: A porcentagem de dados a ser usada para o treinamento.

        Retorna:
            Uma tupla contendo:
            - STATS: Um array com estatísticas de acurácia (média, min, max, mediana, desvio padrão).
            - TX_OK: Um array com a taxa de acerto de cada rodada.
            - last_run_info: Um dicionário com os parâmetros da última execução.
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
            
            if len(np.unique(self.y_train)) != len(np.unique(y_data)):
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


class DMC(BaseClassifier):
    """Classificador de Distância para o Centroide Médio (DMC)."""
    
    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Calcula o centroide médio para cada classe."""
        self.model_params['centroids'] = {}
        self.model_params['class_labels'] = np.unique(y_train)
        for k in self.model_params['class_labels']:
            self.model_params['centroids'][k] = np.mean(X_train[y_train == k], axis=0)

    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prevê com base na menor distância Euclidiana aos centroides da classe."""
        centroids = self.model_params['centroids']
        class_labels = self.model_params['class_labels']
        distances = np.zeros((X_test.shape[0], len(class_labels)))
        
        for i, k in enumerate(class_labels):
            distances[:, i] = np.linalg.norm(X_test - centroids[k], axis=1)
            
        predicted_indices = np.argmin(distances, axis=1)
        return class_labels[predicted_indices]


class NN1(BaseClassifier):
    """Classificador 1-Vizinho Mais Próximo (1-NN) com implementação vetorizada."""

    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Armazena os dados de treino para serem usados durante a predição."""
        self.model_params['X_train'] = X_train
        self.model_params['y_train'] = y_train

    def _predict(self, X_test: np.ndarray) -> np.ndarray:
        """Encontra o vizinho mais próximo para cada ponto de teste e atribui seu rótulo."""
        X_train = self.model_params['X_train']
        y_train = self.model_params['y_train']
        y_pred = np.empty(X_test.shape[0], dtype=y_train.dtype)

        for i, x_test_point in enumerate(X_test):
            differences = X_train - x_test_point
            squared_differences = differences ** 2
            squared_distances = np.sum(squared_differences, axis=1)
            nearest_neighbor_index = np.argmin(squared_distances)
            y_pred[i] = y_train[nearest_neighbor_index]

        return y_pred


class MaxCorr(BaseClassifier):
    """Classificador de Máxima Correlação (MaxCorr)."""

    def _train(self, X_train, y_train):
        """Calcula o centroide médio normalizado para cada classe."""
        self.model_params['norm_centroids'] = {}
        self.model_params['class_labels'] = np.unique(y_train)
        for k in self.model_params['class_labels']:
            centroid = np.mean(X_train[y_train == k], axis=0)
            norm = np.linalg.norm(centroid)
            self.model_params['norm_centroids'][k] = centroid / (norm + 1e-9)

    def _predict(self, X_test):
        """Prevê com base na correlação máxima com os centroides normalizados."""
        norm_centroids = self.model_params['norm_centroids']
        class_labels = self.model_params['class_labels']
        
        norms_X_test = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_test_norm = X_test / (norms_X_test + 1e-9)
        
        correlations = np.zeros((X_test.shape[0], len(class_labels)))
        
        for i, k in enumerate(class_labels):
            correlations[:, i] = X_test_norm @ norm_centroids[k]
        
        predicted_indices = np.argmax(correlations, axis=1)
        return class_labels[predicted_indices]


class QDA(BaseClassifier):
    """Classificador de Análise Discriminante Quadrática (QDA) e suas variantes."""

    def __init__(self, regularization_param=None, mode='QDA'):
        """Inicializa o classificador QDA.

        Parâmetros:
            regularization_param: Parâmetro para os modos de regularização
                                  como 'Tikhonov' ou 'Friedman'.
            mode: O modo de operação. Pode ser 'QDA', 'Tikhonov', 'LDA',
                  'Friedman', ou 'NaiveBayes'.
        """
        super().__init__()
        self.regularization_param = regularization_param
        self.mode = mode
        
    def _train(self, X_train, y_train):
        """Calcula as médias, priores e matrizes de covariância de cada classe."""
        num_features = X_train.shape[1]
        self.model_params['class_labels'] = np.unique(y_train)
        self.model_params['covariances'] = {}
        self.model_params['means'] = {}
        self.model_params['inv_covs'] = {}
        self.model_params['log_dets'] = {}
        self.model_params['log_priors'] = {}

        pooled_cov = np.zeros((num_features, num_features))
        friedman_raw_covs = {}

        for k in self.model_params['class_labels']:
            X_k = X_train[y_train == k]
            n_k = X_k.shape[0]
            prior = n_k / len(y_train)
            
            self.model_params['means'][k] = np.mean(X_k, axis=0)
            self.model_params['log_priors'][k] = np.log(prior)
            
            cov_k = np.cov(X_k, rowvar=False)

            if self.mode == 'Tikhonov':
                cov_k += np.eye(num_features) * self.regularization_param
            elif self.mode in ['LDA', 'Friedman']:
                pooled_cov += cov_k * prior
                if self.mode == 'Friedman':
                    friedman_raw_covs[k] = cov_k
            elif self.mode == 'NaiveBayes':
                cov_k = np.diag(np.diag(cov_k))
            
            self.model_params['covariances'][k] = cov_k

            try:
                self.model_params['inv_covs'][k] = np.linalg.inv(cov_k)
            except np.linalg.LinAlgError:
                self.model_params['inv_covs'][k] = np.linalg.pinv(cov_k)

            sign, log_det = np.linalg.slogdet(cov_k)
            self.model_params['log_dets'][k] = log_det

        if self.mode == 'LDA':
            sign, log_det = np.linalg.slogdet(pooled_cov)
            try:
                inv_pooled_cov = np.linalg.inv(pooled_cov)
            except np.linalg.LinAlgError:
                inv_pooled_cov = np.linalg.pinv(pooled_cov)

            for k in self.model_params['class_labels']:
                self.model_params['covariances'][k] = pooled_cov
                self.model_params['inv_covs'][k] = inv_pooled_cov
                self.model_params['log_dets'][k] = 0

        if self.mode == 'Friedman':
            for k in self.model_params['class_labels']:
                X_k = X_train[y_train == k]
                n_k = X_k.shape[0]
                c = friedman_raw_covs[k]
                lambda_param = self.regularization_param
                
                cov_k_num = (1 - lambda_param) * (n_k * c) + lambda_param * (pooled_cov * len(X_train))
                cov_k_den = (1 - lambda_param) * n_k + lambda_param * len(X_train)
                cov_k = cov_k_num / cov_k_den
                
                self.model_params['covariances'][k] = cov_k

                try:
                    self.model_params['inv_covs'][k] = np.linalg.inv(cov_k)
                except np.linalg.LinAlgError:
                    self.model_params['inv_covs'][k] = np.linalg.pinv(cov_k)

                sign, log_det = np.linalg.slogdet(cov_k)
                self.model_params['log_dets'][k] = log_det

    def _predict(self, X_test):
        """Prevê a classe que minimiza a função discriminante quadrática."""
        class_labels = self.model_params['class_labels']
        scores = np.zeros((X_test.shape[0], len(class_labels)))

        for i, k in enumerate(class_labels):
            diff = X_test - self.model_params['means'][k]
            mahalanobis = np.sum((diff @ self.model_params['inv_covs'][k]) * diff, axis=1)
            scores[:, i] = mahalanobis + self.model_params['log_dets'][k] - 2 * self.model_params['log_priors'][k]

        predicted_indices = np.argmin(scores, axis=1)
        return class_labels[predicted_indices]

    def evaluate(self, data: np.ndarray, num_runs: int, train_percent: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Sobrescreve 'evaluate' para adicionar o posto da covariância aos resultados."""
        stats, accuracy_rates, last_run_info = super().evaluate(data, num_runs, train_percent)
        
        self.model_params['covariances_rank'] = {}
        for i in self.model_params['covariances']:
            self.model_params['covariances_rank'][i] = np.linalg.matrix_rank(self.model_params['covariances'][i])
            
        last_run_info["model_params"] = self.model_params

        return stats, accuracy_rates, last_run_info