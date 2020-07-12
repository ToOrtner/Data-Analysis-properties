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

# Dropeo las huertas porque solo hay una
train_df = train_df.drop(train_df[train_df['tipodepropiedad'] == 'Huerta'].index)
# Dropeo las Lote porque solo hay una
train_df = train_df.drop(train_df[train_df['tipodepropiedad'] == 'Lote'].index)
# Dropeo las quintas vacacionales porque solo los precios son cualquiera
train_df = train_df.drop(train_df[train_df['tipodepropiedad'] == 'Quinta Vacacional'].index)
# Dropeo los ranchos porque solo los precios son cualquiera
train_df = train_df.drop(train_df[train_df['tipodepropiedad'] == 'Rancho'].index)

predict_column = 'precio'
carititud_column = "carititud"
segments = ['tipodepropiedad', 'usosmultiples', 'banos']
text_features = ['titulo', 'descripcion']
features = ['metrostotales', 'metroscubiertos', 'garages']

model1 = Model(train_df, features=features, segment_columns=segments)

model1.regresionar()

print("""---------------------------------------
Scores para modelo sin NLP
---------------------------------------""")
print(model1.scores_por_segmento())
print(model1.prom_scores())

nlpModel = NlpModel(train_df, text_features=text_features, features=features, segment_columns=segments)

nlpModel.regresionar()

print("""---------------------------------------
Scores para modelo con NLP
---------------------------------------""")

print(nlpModel.scores_por_segmento())
print(nlpModel.prom_scores())
