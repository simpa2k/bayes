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
#include "../catch.h"
#include <armadillo>

#include "../../functions.h"

using namespace std;
using namespace arma;

shared_ptr<umat> gatherVisibleData(umat& hiddenData, vector<mat>& thetaVisible, mt19937& engine) {

    umat visibleData;
    for (auto &&node : thetaVisible) {
        umat nodeData = *simulateVisibleData(hiddenData, node, engine);
        visibleData = join_rows(visibleData, nodeData);
    }

    return make_shared<umat>(visibleData);
}

TEST_CASE("Entire use case", "[bayes]") {

    const int SAMPLES = 20000;
    const int LEARNING_ITERATIONS = 1000;

    std::random_device r;
    std::mt19937 engine(r());

    arma::mat thetaHidden = {0.30, 0.25, 0.45};
    arma::mat e0 = {
            {0.15, 0.65, 0.13},
            {0.50, 0.15, 0.57}
    };

    arma::mat e1 = {
            {0.50, 0.75, 0.40},
            {0.40, 0.05, 0.35},
            {0.10, 0.20, 0.25}
    };

    arma::mat e2 = {
            {0.24, 0.42, 0.34},
            {0.66, 0.38, 0.56},
            {0.10, 0.20, 0.10}
    };

    arma::mat e3 = {
            {0.13, 0.52, 0.85},
            {0.67, 0.28, 0.05},
            {0.20, 0.20, 0.10}
    };

    arma::mat e4 = {
            {0.42, 0.56, 0.20},
            {0.38, 0.34, 0.70}
    };

    arma::mat e5 = {
            {0.42, 0.43, 0.12},
            {0.28, 0.47, 0.78},
    };

    arma::mat e6 = {
            {0.42, 0.60, 0.25},
            {0.48, 0.30, 0.55},
            {0.10, 0.10, 0.20}
    };

    arma::mat e7 = {
            {0.52, 0.56, 0.36},
            {0.28, 0.24, 0.44},
            {0.20, 0.10, 0.20}
    };

    arma::mat e8 = {
            {0.42, 0.32, 0.30},
            {0.38, 0.58, 0.50},
    };

    /*
     * Generate data for the hidden node.
     */
    shared_ptr<umat> hiddenData = simulateHiddenData(thetaHidden, SAMPLES, engine);

    /*
     * Generate data for the visible nodes based on the hidden data.
     */
    vector<mat> thetaVisible = {e0, e1, e2, e3, e4, e5, e6, e7, e8};
    shared_ptr<umat> visibleData = gatherVisibleData(*hiddenData, thetaVisible, engine);

    std::cout << "Before randomization: " << *computeThetaHidden(*hiddenData) << std::endl;

    /*
     * Obscure hidden node.
     */
    std::uniform_int_distribution<> dist(0, 2);
    hiddenData->imbue([&] () { return dist(engine); });

    std::cout << "After randomization: " << *computeThetaHidden(*hiddenData) << std::endl;

    /*
     * Re-learn hidden distribution.
     */
    setEngine(engine);
    shared_ptr<mat> thetaLearned = learn(*hiddenData, *visibleData, LEARNING_ITERATIONS);
    
    std::cout << "After computation: " << *thetaLearned << std::endl;
}