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

#include "../../classifier/functions.h"
#include "../../classifier/Classifier.h"

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
    auto engine = make_shared<mt19937>(r());

    Classifier classifier(engine);

    mat thetaHidden = {0.30, 0.25, 0.45};
    mat v0 = {
            {0.15, 0.65, 0.13},
            {0.50, 0.15, 0.57}
    };

    mat v1 = {
            {0.50, 0.75, 0.40},
            {0.40, 0.05, 0.35},
            {0.10, 0.20, 0.25}
    };

    mat v2 = {
            {0.24, 0.42, 0.34},
            {0.66, 0.38, 0.56},
            {0.10, 0.20, 0.10}
    };

    mat v3 = {
            {0.13, 0.52, 0.85},
            {0.67, 0.28, 0.05},
            {0.20, 0.20, 0.10}
    };

    mat v4 = {
            {0.42, 0.56, 0.20},
            {0.38, 0.34, 0.70}
    };

    mat v5 = {
            {0.42, 0.43, 0.12},
            {0.28, 0.47, 0.78},
    };

    mat v6 = {
            {0.42, 0.60, 0.25},
            {0.48, 0.30, 0.55},
            {0.10, 0.10, 0.20}
    };

    mat v7 = {
            {0.52, 0.56, 0.36},
            {0.28, 0.24, 0.44},
            {0.20, 0.10, 0.20}
    };

    mat v8 = {
            {0.42, 0.32, 0.30},
            {0.38, 0.58, 0.50},
    };

    setEngine(*engine);

    /*
     * Generate data for the hidden node.
     */
    shared_ptr<umat> hiddenData = simulateHiddenData(thetaHidden, SAMPLES, *engine);

    /*
     * Generate data for the visible nodes based on the hidden data.
     */
    vector<mat> thetaVisible = {v0, v1, v2, v3, v4, v5, v6, v7, v8};
    shared_ptr<umat> visibleData = gatherVisibleData(*hiddenData, thetaVisible, *engine);

    std::cout << "Before randomization: " << *computeThetaHidden(*hiddenData) << std::endl;

    SECTION("Unknown hidden data") {

        /*
         * Obscure hidden node.
         */
        std::uniform_int_distribution<> dist(0, 2);
        hiddenData->imbue([&] () { return dist(*engine); });

        std::cout << "After randomization: " << *computeThetaHidden(*hiddenData) << std::endl;

        /*
         * Re-learn hidden distribution.
         */
        shared_ptr<mat> thetaLearned = classifier.learn(*hiddenData, *visibleData, LEARNING_ITERATIONS);
        std::cout << "After computation: " << *thetaLearned << std::endl;
    }

    SECTION("Known hidden data") {

        /*
         * Set size of training and evaluation datasets.
         */
        uword trainingSetSize = SAMPLES * 0.9;
        umat hiddenTrainingSet = hiddenData->head_cols(trainingSetSize);
        umat visibleTrainingSet = visibleData ->head_rows(trainingSetSize);

        uword evaluationSetSize = SAMPLES - trainingSetSize;
        umat hiddenEvaluationSet = hiddenData->tail_cols(evaluationSetSize);
        umat visibleEvaluationSet = visibleData->tail_rows(evaluationSetSize);

        /*
         * Learn hidden distribution.
         */
        shared_ptr<mat> thetaLearned = classifier.learn(hiddenTrainingSet, visibleTrainingSet, 1);
        std::cout << "After computation: " << *thetaLearned << std::endl;

        /*
         * Evaluate
         */
        shared_ptr<vector<mat>> thetaVisible = computeThetaVisible(hiddenTrainingSet, visibleTrainingSet);
        shared_ptr<mat> thetaGuessed = classifier.imputeHiddenNode(visibleEvaluationSet,
                                                                   *thetaLearned,
                                                                   *thetaVisible,
                                                                   false);

        int correctGuesses = 0;

        for (int i = 0; i < thetaGuessed->n_rows; ++i) {

            rowvec row = thetaGuessed->row(i);

            row = row / accu(row); // Normalize
            int maxIndex = 0;

            /*
             * Pick out most likely classification.
             */
            for (int j = 0; j < row.n_cols; ++j) {
                if (row.col(j)(0) > row.col(maxIndex)(0)) {
                    maxIndex = j;
                }
            }

            if (maxIndex == hiddenEvaluationSet.col(i)(0)) {
                ++correctGuesses;
            }
        }

        double correctnessPercentage = (double(correctGuesses) / evaluationSetSize) * 100;
        std::cout << "Guessed correctly " << correctnessPercentage << "% of the time." << std::endl;
    }
}