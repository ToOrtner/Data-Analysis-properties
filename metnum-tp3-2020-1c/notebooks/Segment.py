import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class Segment:
    def __init__(self, name, linear_regressor, features, metrics, k=5):
        self.name = name
        self.linear_regressor = linear_regressor
        self.features = features
        self.scores = []
        self.metrics = metrics
        self.k = k 
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, A, reales):
        reales_scaled = self.scaler.fit_transform(reales.reshape(-1, 1)).reshape(-1,)
        self._cv(A, reales_scaled)
        self.linear_regressor.fit(A, reales_scaled)

    def predict(self, A):
        pred = self.linear_regressor.predict(A)
        return self.scaler.inverse_transform(pred).reshape(-1)

    def _cv(self, A, reales):
        
        self.scores = np.zeros(len(self.metrics))

        kf = KFold(n_splits=self.k)
        for train_index, test_index in kf.split(A):
            # Divido en datos de entrenamiento y testeo
            A_train, A_test = A[train_index], A[test_index]
            reales_train, reales_test = reales[train_index], reales[test_index]
            reales_test = self.scaler.inverse_transform(reales_test).reshape(-1)

            # Fiteo con los de entrenamiento
            self.linear_regressor.fit(A_train, reales_train)

            # Predigo con los de testeo
            predicted = self.predict(A_test)
            
            # Calculo los scores parciales de este fold
            partial_scores = [0] * len(self.metrics)
            try:
                partial_scores = self._calc_metrics(reales_test, predicted)
            except ValueError as err:
                print(f'Error calculando metricas en segmento {self.name}:\n{err}')
                
            self.scores += partial_scores
            
        # Promedio los scores
        self.scores /= self.k
        
    def _calc_metrics(self, X, y):
        return [m(X, y) for m in self.metrics]
    
    def get_df_scores(self):
        df = pd.Dataframe()
        for score, metric in zip(self.scores, self.metrics):
            df[metric._name_] = score
        return df