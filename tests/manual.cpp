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

#define CATCH_CONFIG_MAIN
#include "catch.h"
#include <armadillo>

#include "../functions.h"

TEST_CASE("Entire use case", "[bayes]") {

    const int SAMPLES = 20000;
    const int LEARNING_ITERATIONS = 1000;

    std::random_device r;
    std::mt19937 engine(r());

    arma::mat thetaHidden = {0.30, 0.25, 0.45};
    arma::mat e0 = {
            {0.55, 0.95, 0.11},
            {0.45, 0.05, 0.89}
    };

    arma::mat e1 = {
            {0.60, 0.95, 0.40},
            {0.40, 0.05, 0.60}
    };

    arma::mat e2 = {
            {0.24, 0.42, 0.34},
            {0.76, 0.58, 0.66}
    };

    arma::mat e3 = {
            {0.13, 0.72, 0.95},
            {0.87, 0.28, 0.05}
    };

    arma::mat e4 = {
            {0.62, 0.66, 0.20},
            {0.38, 0.34, 0.80}
    };

    arma::mat e5 = {
            {0.42, 0.43, 0.12},
            {0.58, 0.57, 0.88}
    };

    arma::mat e6 = {
            {0.42, 0.60, 0.25},
            {0.48, 0.40, 0.75}
    };

    arma::mat e7 = {
            {0.72, 0.76, 0.36},
            {0.28, 0.24, 0.64}
    };

    arma::mat e8 = {
            {0.62, 0.32, 0.30},
            {0.38, 0.68, 0.70}
    };

    arma::umat dataHidden = simulateHiddenData(thetaHidden, SAMPLES, &engine);

    std::shared_ptr<arma::umat> dataE0 = simulateVisibleData(dataHidden, e0, &engine);
    std::shared_ptr<arma::umat> dataE1 = simulateVisibleData(dataHidden, e1, &engine);
    std::shared_ptr<arma::umat> dataE2 = simulateVisibleData(dataHidden, e2, &engine);
    std::shared_ptr<arma::umat> dataE3 = simulateVisibleData(dataHidden, e3, &engine);
    std::shared_ptr<arma::umat> dataE4 = simulateVisibleData(dataHidden, e4, &engine);
    std::shared_ptr<arma::umat> dataE5 = simulateVisibleData(dataHidden, e5, &engine);
    std::shared_ptr<arma::umat> dataE6 = simulateVisibleData(dataHidden, e6, &engine);
    std::shared_ptr<arma::umat> dataE7 = simulateVisibleData(dataHidden, e7, &engine);
    std::shared_ptr<arma::umat> dataE8 = simulateVisibleData(dataHidden, e8, &engine);

    arma::umat dataVisible = arma::join_rows(*dataE0, *dataE1);
    dataVisible = arma::join_rows(dataVisible, *dataE2);
    dataVisible = arma::join_rows(dataVisible, *dataE3);
    dataVisible = arma::join_rows(dataVisible, *dataE4);
    dataVisible = arma::join_rows(dataVisible, *dataE5);
    dataVisible = arma::join_rows(dataVisible, *dataE6);
    dataVisible = arma::join_rows(dataVisible, *dataE7);
    dataVisible = arma::join_rows(dataVisible, *dataE8);

    std::shared_ptr<std::vector<arma::mat>> computedThetaVisible = computeThetaVisible(dataHidden, dataVisible);

    std::cout << "Före slumpning: " << computeThetaHidden(&dataHidden) << std::endl;

    std::uniform_int_distribution<> dist(0, 2);
    dataHidden.imbue([&] () { return dist(engine); });

    std::cout << "Efter slumpning: " << computeThetaHidden(&dataHidden) << std::endl;

    setEngine(engine);
    arma::mat thetaLearned = learn(dataHidden, dataVisible, LEARNING_ITERATIONS);
    
    std::cout << "Efter beräkning: " << thetaLearned << std::endl;
}