import metnum
import numpy as np
from Segment import Segment
from sklearn.metrics import mean_squared_error as RMSE, mean_squared_log_error as RMSLE

class Model:
    def __init__(self, features, segment_columns, kfold=5, predict_column='precio'):
        self.features = features
        self.segment_columns = segment_columns
        self.predict_column = predict_column
        self.segments = np.array([])
        self.kfold = kfold

    def metrics(self):
        return [RMSE, RMSLE]
    
    def error_gral(self):
        error = 0
        if(len(self.segments) == 0):
            print("Primero deberias ejecutar la regresion")
            return -1

        for segment in self.segments:
            error += segment.errors

        return error/len(self.segments)
            

    def regresionar(self, df):
        # Genera una copia para no romper nada fuera del modelo
        df_copy = df.dropna(subset=(self.features + self.segment_columns)).copy()
        self.regresionar_segmentos(df_copy, self.segment_columns)
        return self.segments

    def regresionar_segmentos(self, df, segment_columns, segment_name=''):
        # Si no hay mas columnas de segmentos para achicar el df, regresiono con ese df.
        if len(segment_columns) == 0:
            self.regresionar_segmento(df, segment_name)
            return

        # Separa la columna del ultimo segmento e itera sobre los valores de la misma
        segment_column = segment_columns[-1]
        for name in df[segment_column].unique():
            # Achica el df y las columnas de segmentos restantes, segun los datos del segmento. 
            # Tambien va concatenando los nombres para poder diferenciar los segmentos
            self.regresionar_segmentos(df[df[segment_column] == name], segment_columns[:-1], f'{segment_name}/{name}')

    def regresionar_segmento(self, df_segment, segment_name):
        # Separa la matriz A y el vector de los resultados reales (los precios por ej.) para pasarselos al segmento y que los fitee
        A = df_segment[self.features].values
        reales = df_segment[self.predict_column].values

        # Si NO hay por lo menos k elementos para hacer kfold, este segmento no va a aportar información relevante para el modelo
        if len(reales) >= self.kfold:
            # Creo el segmento
            segment = Segment(segment_name, metnum.LinearRegression(), self.features)
            #print(f'Segemento: {segment_name} elementos \n')
            # Fit y predict
            segment.execute(A, reales, self.metrics(), self.kfold)
            # Guardo el segmento
            self.segments = np.append(self.segments, segment)