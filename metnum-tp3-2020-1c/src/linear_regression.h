#pragma once

#include "types.h"

class LinearRegression {
public:
    LinearRegression();

    void fit(Matrix X, Vector y);

    Matrix predict(Matrix X);

    Vector predictOne(Vector X);
private:
    Vector factors;
};
