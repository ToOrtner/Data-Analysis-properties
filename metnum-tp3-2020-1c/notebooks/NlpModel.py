import pandas as pd
import numpy as np

from Model import Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NlpModel(Model):
    
    _text_feat_column = 'text_feat_column'
    _nlp_feat_column = 'nlp_feat_column'
    
    def __init__(self, df, text_features, features, segment_columns, kfold=5, predict_column='precio', drop_na=True):
        super().__init__(df, features, segment_columns, kfold, predict_column, False, drop_na=False)
        self.text_features = text_features
        self.df[self._text_feat_column] = df[text_features].astype(str).agg(' '.join, axis=1)
        if drop_na:
            self.df = self.df.dropna()
        
    def regresionar(self):
        # Creo la columna con los datos del estimador con NLP
        self._create_nlp_column()
        # Agrego esa columna como una feature mas para usar en la regresion lineal
        self.features.append(self._nlp_feat_column)
        super().regresionar()
        
    def _create_nlp_column(self):
        self.estimator = self._get_estimator()
        X, y = self._get_data_to_fit()
        self.estimator.fit(X, y)
        self.df[self._nlp_feat_column] = self.estimator.predict(X)
        
        
    def _get_estimator(self):
        params = {
            'count__max_features': 5000,
            'count__min_df': 5,
            'desc__n_components': 100,
            'reg__hidden_layer_sizes': (50, 20),
            'reg__max_iter': 50,
            'reg__solver': 'adam'
        }
        
        pipeline = Pipeline([
            ('count', CountVectorizer()),
            ('desc', TruncatedSVD()),
            ('reg', MLPRegressor())
        ], verbose=True)
        
        pipeline.set_params(**params)
        
        return pipeline
    
    def _get_data_to_fit(self):
        X = self.df[self._text_feat_column].values

        # Escalado de datos a predecir
        scaler = StandardScaler(with_mean=False)
        to_predict = self.df[self.predict_column].values.reshape(1,-1)
        y = scaler.fit_transform(to_predict).reshape((-1,))
        
        return (X, y)