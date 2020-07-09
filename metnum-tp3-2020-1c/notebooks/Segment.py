import numpy as np
from sklearn.model_selection import KFold

class Segment:
    def __init__(self, name, linear_regressor, features):
        self.name = name
        self.linear_regressor = linear_regressor
        self.features = features
        self.errors = 0

    def fit(self, A, reales):
        self.linear_regressor.fit(A, reales)

    def predict(self, A):
        return self.linear_regressor.predict(A)

    def execute(self, A, reales, metrics, k=5):

        self.errors = np.zeros_like(metrics)

        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(A):
            # Divido en datos de entrenamiento y testeo
            A_train, A_test = A[train_index], A[test_index]
            reales_train, reales_test = reales[train_index], reales[test_index]
            # Fiteo con los de entrenamiento
            self.fit(A_train, reales_train)

            # Predigo con los de testeo
            aproximados = self.predict(A_test)
            i = 0
            # Calculo y guardo las metricas pasadas por parametro
            for metric in metrics:
                self.errors[i] += metric(reales_test, aproximados)
                i += 1

        self.errors /= k
