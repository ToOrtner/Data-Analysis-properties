import metnum
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pprint import pprint
from Model import Model
from Segment import Segment
from NlpModel import NlpModel
import feats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as RMSE, mean_squared_log_error as RMSLE, balanced_accuracy_score as BAS, \
    make_scorer
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

train_df_original = pd.read_csv('../data/train.csv')
train_df = train_df_original.copy()
# train_df.info()
test_df_original = pd.read_csv('../data/test.csv')
test_df = test_df_original.copy()
# test_df.info()

def drops(df):
    # Dropeo las huertas porque solo hay una
    df = df.drop(df[df['tipodepropiedad'] == 'Huerta'].index)
    # Dropeo las Lote porque solo hay una
    df = df.drop(df[df['tipodepropiedad'] == 'Lote'].index)
    # Dropeo las quintas vacacionales porque solo los precios son cualquiera
    df = df.drop(df[df['tipodepropiedad'] == 'Quinta Vacacional'].index)
    # Dropeo los ranchos porque solo los precios son cualquiera
    df = df.drop(df[df['tipodepropiedad'] == 'Rancho'].index)


# feature engineer
train_df = feats.newfeats(train_df)


predict_column = 'precio'
carititud_column = "carititud"
segments = ['urbana', 'calurosa', 'parachicos']
text_features = ['titulo', 'descripcion']
features = ['metroscubiertos', 'mejorciudad']


def correr(df, segments, text_features, features, predict_column='precio', normal=True, nlp=True):
    to_predict = df.dropna(subset=features + segments + text_features + [predict_column])

    reales = to_predict[predict_column]
    predictedM = np.array([])
    predictedNLP = np.array([])
    if normal:
        model1 = Model(df, features=features, segment_columns=segments)

        model1.regresionar()

        pd.set_option('display.float_format', lambda x: '%.3f' % x)

        df_to_show = pd.DataFrame()

        df_to_show["real"] = reales
        print("reales  ", reales.shape)
        predictedM = model1.predict(to_predict[features + segments])
        print("predicc", predictedM.shape)
        df_to_show["predicted"] = predictedM

        print(df_to_show)

        print("""---------------------------------------
        Scores para modelo sin NLP
        ---------------------------------------""")
        print(model1.scores_por_segmento())
        print(model1.prom_scores())

    if nlp:
        nlpModel = NlpModel(df, text_features=text_features, features=features, segment_columns=segments)

        nlpModel.regresionar()

        print("""---------------------------------------
        Scores para modelo con NLP
        ---------------------------------------""")
        for segment in nlpModel.segments.values():
            print(segment.get_df_scores())
        # print(nlpModel.scores_por_segmento())

        print("""---------------------------------------
        Scores promedio
        ---------------------------------------""")
        print(nlpModel.prom_scores())

        df_to_show = pd.DataFrame()

        df_to_show["real"] = reales
        predictedNLP = nlpModel.predict(to_predict[features + segments + text_features])
        df_to_show["predicted"] = predictedNLP

        print(df_to_show)

    return reales, predictedM, predictedNLP

predic = correr(train_df, segments, text_features, features)
