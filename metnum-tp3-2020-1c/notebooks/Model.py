import metnum
import numpy as np
from Segment import Segment
from sklearn.metrics import mean_squared_error as RMSE, mean_squared_log_error as RMSLE


class Model:
    def __init__(self, features, segment_column, predict_column='precio'):
        self.features = features
        self.segment_column = segment_column
        self.predict_column = predict_column

    def metrics(self):
        return [RMSE, RMSLE]

    def regresionar_segmento(self, df_segment, name):
        A = df_segment[self.features].values
        precios_reales = df_segment[self.predict_column].values
        # Creo el segmento
        segment = Segment(name, metnum.LinearRegression(), self.features)
        # print(f'Segemento: {name} \n')
        # Fit y predict
        segment.execute(A, precios_reales, self.metrics())
        # Guardo el segmento
        self.segments = np.append(self.segments, segment)

    def regresionar(self, df):
        df_copy = df.dropna(subset=(self.features + [self.segment_column])).copy()
        self.segments = np.array([])

        # Lo separe en regresionar_segmento porque no se si despues va a haber mas de un segmento, para que no se haga un choclo

        for name in df_copy[self.segment_column].unique():
            # Leo los datos correspondientes al segmeto
            df_segment = df_copy[df_copy[
                                     self.segment_column] == name]  # Aquellos datos que no contengan la columna, no son tomados en cuenta
            self.regresionar_segmento(df_segment, name)

        return self.segments
