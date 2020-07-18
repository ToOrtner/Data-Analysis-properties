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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Dropeamos los tipos de propiedades que generen problemas con la experimentación
def drops(df):
    # Dropeamos las huertas porque solo hay una
    df = df.drop(df[df['tipodepropiedad'] == 'Huerta'].index)
    # Dropeamos las Lote porque solo hay una
    df = df.drop(df[df['tipodepropiedad'] == 'Lote'].index)
    # Dropeamos las quintas vacacionales porque solo los precios son cualquiera
    df = df.drop(df[df['tipodepropiedad'] == 'Quinta Vacacional'].index)
    # Dropeamos los ranchos porque solo los precios son cualquiera
    df = df.drop(df[df['tipodepropiedad'] == 'Rancho'].index)
