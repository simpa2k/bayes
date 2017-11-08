//
// Created by simon on 2017-11-08.
//

#ifndef BAYES_UTILS_H
#define BAYES_UTILS_H

#include <armadillo>
#include <memory>

using namespace std;
using namespace arma;

void compareMatrices(const umat& expected, const umat& given);
shared_ptr<umat> gatherVisibleData(umat& hiddenData, vector<mat>& thetaVisible, mt19937& engine);

#endif //BAYES_UTILS_H
