#pragma once

#include "types.h"

static double apply_RMSE(const Vector& X, const Vector& Y, double (*f)(double)) {
    int n = X.size();
    // ŷ_i = f'(x_i)
    // Aplico f a cada elemento de Y y consigo
    Vector Y_estimated = X.unaryExpr(f);

    // e_i = y_i − ŷ_i
    Vector error = Y - Y_estimated;

    double error_norm = error.norm();

    // RMSE = sqrt(sum(e_i^2)/n) == sqrt(sum(e_i^2))/sqrt(n) == ||e||2 / sqrt(n)
    return error_norm / sqrt(n);
}


static double apply_RMSLE(const Vector& X, const Vector& Y, double (*f)(double)) {
    int n = X.size();
    // ŷ_i = f'(x_i)
    // Aplico f a cada elemento de Y y consigo
    Vector Y_estimated = X.unaryExpr(f);

    // e_i = log(y_i + 1) − log(ŷ_i + 1)
    Vector error = (Y + Vector(n).setOnes()).unaryExpr(&log) - (Y_estimated + Vector(n).setOnes()).unaryExpr(&log);

    double error_norm = error.norm();

    // RMSLE = sqrt(sum(e_i^2)/n) == sqrt(sum(e_i^2))/sqrt(n) == ||e||2 / sqrt(n)
    return error_norm / sqrt(error.size());
}