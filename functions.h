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
void expandVertically(arma::mat* target, int targetRows);
void expandHorizontally(arma::umat* target, int targetColumns);

arma::umat simulateHiddenData(arma::mat distribution, const int samples, std::mt19937* engine);
std::shared_ptr<arma::umat> simulateVisibleData(arma::umat hiddenData, arma::mat distribution, std::mt19937 *engine);

arma::mat computeThetaHidden(arma::umat* hiddenData);
arma::mat computeThetaVisible(arma::umat* dataHidden, arma::umat* dataVisible);
std::shared_ptr<arma::mat> computeThetaVisibleForNode(const arma::umat& hiddenData, const arma::umat& visibleData);
std::shared_ptr<std::vector<arma::mat>> computeThetaVisible(arma::umat& hiddenData, arma::umat& visibleData);

arma::mat replaceAllValues(arma::umat* dataVisible, int thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible);

arma::mat imputeHiddenNode(arma::umat* dataVisible, arma::mat thetaHidden, std::shared_ptr<std::vector<arma::mat>> thetaVisible, bool generateNewData);
arma::mat learn(arma::umat dataHidden, arma::umat dataVisible, int learningIterations);

#endif //BAYES_MAIN_H_H
