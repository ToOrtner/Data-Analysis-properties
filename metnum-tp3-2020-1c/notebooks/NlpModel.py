from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Model import Model


class NlpModel(Model):
    _text_feat_column = 'text_feat_column'
    _nlp_feat_column = 'nlp_feat_column'

    def __init__(self, df, text_features, features, segment_columns, kfold=5, predict_column='precio', drop_na=True):
        super().__init__(df, features, segment_columns, kfold, predict_column, False, drop_na=False)
        self.text_features = text_features.copy()
        self.df[self._text_feat_column] = self._join_text_features(df)
        self.scaler = StandardScaler(with_mean=False)
        if drop_na:
            self.df = self.df.dropna()
        self.is_nlp_fitted = False
        self.estimator = self._get_estimator()

    def regresionar(self):
        if not self.is_nlp_fitted:
            self.fit_nlp()
        super().regresionar()

    def predict(self, df):
        if df[self.text_features].isnull().values.any():
            raise ValueError(
                f'El dataframe tiene que tener las columnas '
                f'{self.text_features} libre de NaN\'s'
            )
        # Copio porque voy a modificar el dataframe
        df_copy = df.copy()
        df_copy[self._text_feat_column] = self._join_text_features(df_copy)
        X = df_copy[self._text_feat_column].values
        self._create_nlp_column(df_copy, X)
        super().predict(df_copy)

    def fit_nlp(self):
        # Entreno al estimador de nlp
        X, y = self._get_data_to_fit()
        self.estimator.fit(X, y)
        # Creo la columna con los datos del estimador con NLP
        self._create_nlp_column(self.df, X)
        # Agrego esa columna como una feature mas para usar en la regresion lineal
        self.features.append(self._nlp_feat_column)
        self.is_nlp_fitted = True

    def _create_nlp_column(self, df, X):
        df[self._nlp_feat_column] = self.estimator.predict(X)

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
        to_predict = self.df[self.predict_column].values.reshape(-1, 1)
        y = self.scaler.fit_transform(to_predict).reshape(-1)

        return X, y

    def _join_text_features(self, df):
        return df[self.text_features].astype(str).agg(' / '.join, axis=1)
