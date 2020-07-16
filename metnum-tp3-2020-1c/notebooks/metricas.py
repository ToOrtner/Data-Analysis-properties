from sklearn.metrics import mean_squared_error as RMSE, mean_squared_log_error as RMSLE,\
    r2_score as R2_SCORE, max_error as MAX_ERROR, mean_absolute_error as MAE
import numpy as np

def NRMSE(original, predicho):
    prom = np.mean(original)
    return RMSE(original, predicho) / prom

def porcentajeProm (original, predicho):

    return sum(abs(original - predicho) * 100 / original) / original.size
