#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include "linear_regression.h"
#include "metrics.hpp"


using namespace std;
namespace py = pybind11;

LinearRegression::LinearRegression() {
}

void LinearRegression::fit(Matrix X, Vector y) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Vector(X.rows()).setOnes();

    factors = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}


Matrix LinearRegression::predict(Matrix X) {
    if (factors.size() == 0)
        throw runtime_error("Primero tenes que correr el fit");
    return X * factors;
}