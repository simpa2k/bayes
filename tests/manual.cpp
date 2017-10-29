//
// Created by simon on 2017-10-29.
//

#define CATCH_CONFIG_MAIN
#include "catch.h"
#include <armadillo>

#include "../functions.h"

TEST_CASE("Entire use case", "[bayes") {

    const int SAMPLES = 10000;
    const int LEARNING_ITERATIONS = 400;

    std::random_device r;
    std::mt19937 engine(r());

    arma::mat thetaHidden = {0.15, 0.85};
    arma::mat e0 = {
            {0.55, 0.95},
            {0.45, 0.05}
    };

    arma::mat e1 = {
            {0.60, 0.95},
            {0.40, 0.05}
    };

    arma::mat e2 = {
            {0.24, 0.42},
            {0.76, 0.58}
    };

    arma::mat e3 = {
            {0.13, 0.72},
            {0.87, 0.28}
    };

    arma::mat e4 = {
            {0.62, 0.66},
            {0.38, 0.34}
    };
    /*arma::mat thetaVisible = {
            {0.55, 0.95},
            {0.60, 0.95},
            {0.24, 0.42},
            {0.13, 0.72},
            {0.62, 0.66}
    };*/

    //std::vector<arma::mat> thetaVisible = {e0, e1, e2, e3, e4};

    arma::umat dataHidden = simulateHiddenData(thetaHidden, SAMPLES, &engine);

    std::shared_ptr<arma::umat> dataE0 = simulateVisibleData(dataHidden, e0, &engine);
    std::shared_ptr<arma::umat> dataE1 = simulateVisibleData(dataHidden, e1, &engine);
    std::shared_ptr<arma::umat> dataE2 = simulateVisibleData(dataHidden, e2, &engine);
    std::shared_ptr<arma::umat> dataE3 = simulateVisibleData(dataHidden, e3, &engine);
    std::shared_ptr<arma::umat> dataE4 = simulateVisibleData(dataHidden, e4, &engine);

    arma::umat dataVisible = arma::join_rows(*dataE0, *dataE1);
    dataVisible = arma::join_rows(dataVisible, *dataE2);
    dataVisible = arma::join_rows(dataVisible, *dataE3);
    dataVisible = arma::join_rows(dataVisible, *dataE4);

    std::shared_ptr<std::vector<arma::mat>> computedThetaVisible = computeThetaVisible(dataHidden, dataVisible);

    //arma::umat dataVisible = simulateVisibleData(arma::trans(dataHidden), thetaVisible, SAMPLES);
    //arma::umat dataVisible = simulateVisibleData(dataHidden, thetaVisible, &engine);

    //arma::mat computedThetaHidden = computeThetaHidden(&dataHidden);
    //arma::mat computedThetaVisible = computeThetaVisible(&dataHidden, &dataVisible);

    dataHidden.imbue([] () { return rand() % 2; });

    arma::mat thetaLearned = learn(dataHidden, dataVisible, LEARNING_ITERATIONS);

    std::cout << thetaLearned << std::endl;
}