//
// Created by Simon Olofsson on 2017-11-05.
//

#ifndef BAYES_CLASSIFIER_H
#define BAYES_CLASSIFIER_H

#include <armadillo>

using namespace std;
using namespace arma;

class Classifier {

    shared_ptr<mt19937> engine;

    shared_ptr<mat> computeThetaHidden(const umat& hiddenData);
    shared_ptr<mat> computeThetaVisibleForNode(const umat& hiddenData, int hiddenStates, const umat& visibleData);
    shared_ptr<vector<mat>> computeThetaVisible(umat& hiddenData, umat& visibleData);

    shared_ptr<mat> replaceAllValues(umat& dataVisible, int thetaHidden, vector<mat>& thetaVisible);

public:

    explicit Classifier(shared_ptr<mt19937> engine);

    shared_ptr<mat> imputeHiddenNode(umat& dataVisible, mat& thetaHidden, vector<mat>& thetaVisible, bool generateNewData);
    shared_ptr<mat> learn(umat& dataHidden, umat& dataVisible, int learningIterations);

};


#endif //BAYES_CLASSIFIER_H
