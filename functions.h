//
// Note that this code is partly a translation of a Python example presented in an excellent blog
// post by Jace Kohlmeier at:
// http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy
//
// Apart from the translation, this code expands on the example by providing functionality for using
// an arbitrary amount of visible and hidden states.
//
// All credit goes to Jace Kohlmeier for the original algorithm.
//

#ifndef BAYES_MAIN_H_H
#define BAYES_MAIN_H_H

#include <armadillo>
#include <memory>

using namespace std;
using namespace arma;

void setEngine(mt19937 &eng);

shared_ptr<umat> simulateHiddenData(arma::mat& distribution, const int samples, std::mt19937& engine);
std::shared_ptr<arma::umat> simulateVisibleData(const arma::umat& hiddenData, const arma::mat& distribution, std::mt19937& engine);

shared_ptr<mat> computeThetaHidden(const arma::umat& hiddenData);
arma::mat computeThetaVisible(const arma::umat& dataHidden, const arma::umat& dataVisible);
shared_ptr<mat> computeThetaVisibleForNode(const arma::umat& hiddenData, int hiddenStates, const arma::umat& visibleData);
shared_ptr<vector<arma::mat>> computeThetaVisible(arma::umat& hiddenData, arma::umat& visibleData);

shared_ptr<mat> replaceAllValues(arma::umat& dataVisible, int thetaHidden, vector<mat>& thetaVisible);

shared_ptr<mat> imputeHiddenNode(arma::umat& dataVisible, arma::mat& thetaHidden, vector<mat>& thetaVisible, bool generateNewData);
shared_ptr<mat> learn(arma::umat& dataHidden, arma::umat& dataVisible, int learningIterations);

#endif //BAYES_MAIN_H_H
