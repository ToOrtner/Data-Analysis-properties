import metnum
import numpy as np
from Segment import Segment
from sklearn.metrics import mean_squared_error as RMSE, mean_squared_log_error as RMSLE, r2_score as R2_SCORE, max_error as MAX_ERROR
from scipy import stats

class Model:
    def __init__(self, df, features, segment_columns, kfold=5, predict_column='precio', remove_outliers=True):
        self.features = features
        self.segment_columns = segment_columns
        self.predict_column = predict_column
        self.segments = np.array([])
        self.kfold = kfold
        # Genera una copia para no romper nada fuera del modelo
        self.df = df[self.features + self.segment_columns + [predict_column]].copy()
        self.df = self.df.dropna()
        if remove_outliers:
            self._remove_segment_outliers()

    def metrics(self):
        return [RMSE, RMSLE, R2_SCORE, MAX_ERROR]
    
    def error_gral(self):
        error = 0
        if(len(self.segments) == 0):
            print("Primero deberias ejecutar la regresion")
            return -1

        for segment in self.segments:
            error += segment.errors

        return error/len(self.segments)

    def regresionar(self):
        self._regresionar_segmentos(self.df, 0)
        return self.segments

    def _regresionar_segmentos(self, df, segment_index, segment_name=''):
        # Si no hay mas columnas de segmentos para achicar el df, regresiono con ese df.
        if segment_index == len(self.segment_columns):
            self._regresionar_segmento(df, segment_name)
            return

        # Separa la columna del ultimo segmento e itera sobre los valores de la misma
        segment_column = self.segment_columns[segment_index]
        for name in df[segment_column].unique():
            # Achica el df y las columnas de segmentos restantes, segun los datos del segmento. 
            # Tambien va concatenando los nombres para poder diferenciar los segmentos
            self._regresionar_segmentos(df[df[segment_column] == name], segment_index + 1, f'{segment_name}/{name}')

    def _regresionar_segmento(self, df_segment, segment_name):
        # Separa la matriz A y el vector de los resultados reales (los precios por ej.) para pasarselos al segmento y que los fitee
        A = df_segment[self.features].values
        reales = df_segment[self.predict_column].values

        # Si NO hay por lo menos k elementos para hacer kfold, este segmento no va a aportar información relevante para el modelo
        if len(reales) < self.kfold:
            raise RuntimeError(f'{segment_name} tiene menos de {self.kfold} elementos')
            
        # Creo el segmento
        segment = Segment(str(segment_name), metnum.LinearRegression(), self.features)
        #print(f'Segemento: {segment_name} elementos \n')
        # Fit y predict
        segment.execute(A, reales, self.metrics(), self.kfold)
        # Guardo el segmento
        self.segments = np.append(self.segments, segment)

    def _remove_segment_outliers(self):
        df_features = self.df[self.features]
        # Calc z-score
        z_scores = stats.zscore(df_features)
        abs_z_scores = np.abs(z_scores)

        # Remove where abs z-score is greater or equal to 3 
        filtered_entries = (abs_z_scores < 2).all(axis=1)
        
        self.df = self.df[filtered_entries]

    def _find_segment(self, row):
        path = ''
        for segment_column in self.segment_columns:
            path = f'{path}/{row[segment_column]}'

        return [s for s in self.segments if s.name == path]
    
    def predict_rows(self, df_rows):
        if len(self.segments) == 0:
            print("Primero tenes que regresionar")
            return

        predictions = np.zeros(shape=(len(df_rows),))
        i = 0
        for index, row in df_rows.iterrows():
            segment = self._find_segment(row)
            if(len(segment) == 0):
                print(f'No existe segmento para la entrada número {index}')
            predict = segment[0].predict([row[self.features].values])
            predictions[i] = predict[0][0]
            i +=1

        return predictions