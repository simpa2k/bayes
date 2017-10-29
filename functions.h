//
// Created by simon on 2017-10-29.
//

#ifndef BAYES_MAIN_H_H
#define BAYES_MAIN_H_H

#include <armadillo>
#include <memory>

void expandVertically(arma::mat* target, int targetRows);
void expandHorizontally(arma::umat* target, int targetColumns);

arma::umat simulateHiddenData(arma::mat distribution, const int samples, std::mt19937* engine);
//arma::umat simulateVisibleData(arma::umat dataHidden, arma::mat thetaVisible, const int samples);
std::shared_ptr<arma::umat> simulateVisibleData(arma::umat hiddenData, arma::mat distribution, std::mt19937 *engine);

arma::mat computeThetaHidden(arma::umat* hiddenData);
arma::mat computeThetaVisible(arma::umat* dataHidden, arma::umat* dataVisible);
std::shared_ptr<arma::mat> computeThetaVisibleForNode(const arma::umat& hiddenData, const arma::umat& visibleData);
std::shared_ptr<std::vector<arma::mat>> computeThetaVisible(arma::umat& hiddenData, arma::umat& visibleData);

arma::mat replaceAllValues(arma::umat* dataVisible, int thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible);

//arma::mat imputeHiddenNode(arma::umat* dataVisible, arma::mat thetaHidden, arma::mat thetaVisible, bool generateNewData);
arma::mat imputeHiddenNode(arma::umat* dataVisible, arma::mat thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible, bool generateNewData);
arma::mat learn(arma::umat dataHidden, arma::umat dataVisible, int learningIterations);

#endif //BAYES_MAIN_H_H
