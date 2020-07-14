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
#segments = ['tipodepropiedad', 'usosmultiples', 'banos']
segments = ['urbana', 'banos']
text_features = ['titulo', 'descripcion']
features = ['metrostotales', 'metroscubiertos', 'garages']

#feats obtenidos por feature engineer

#nuevos_feats = ['calurosa', 'parachicos']
train_df = feats.newfeats(train_df)

model1 = Model(train_df, features=features, segment_columns=segments)

model1.regresionar()

pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_to_show = pd.DataFrame()
to_predict = train_df[features + segments + [predict_column]].dropna()[:1000]
df_to_show["real"] = to_predict[predict_column]
df_to_show["predicted"] = model1.predict(to_predict[features + segments])

print(df_to_show)


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
for segment in nlpModel.segments.values():
    print(segment.get_df_scores())
#print(nlpModel.scores_por_segmento())

print("""---------------------------------------
Scores promedio
---------------------------------------""")
print(nlpModel.prom_scores())

df_to_show = pd.DataFrame()
to_predict = train_df[features + segments + text_features + [predict_column]].dropna()[:1000]
df_to_show["real"] = to_predict[predict_column]
df_to_show["predicted"] = nlpModel.predict(to_predict[features + segments + text_features])

print(df_to_show)
