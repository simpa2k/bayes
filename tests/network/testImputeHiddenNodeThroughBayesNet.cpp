#include "../catch.h"
#include "../utils/utils.h"
#include "../../classifier/functions.h"
#include "../../network/BayesianNetwork.h"

using namespace std;
using namespace arma;

TEST_CASE("Test impute hidden node", "[bayes]") {

    const int SAMPLES = 20000;

    random_device r;
    auto engine = make_shared<mt19937>(r());

    BayesianNetwork bayesianNetwork(engine);

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

    bayesianNetwork.add("v0");
    bayesianNetwork.record("v0", trans(visibleData->col(0)));

    bayesianNetwork.add("v1");
    bayesianNetwork.record("v1", trans(visibleData->col(1)));

    bayesianNetwork.add("v2");
    bayesianNetwork.record("v2", trans(visibleData->col(2)));

    bayesianNetwork.add("v3");
    bayesianNetwork.record("v3", trans(visibleData->col(3)));

    bayesianNetwork.add("v4");
    bayesianNetwork.record("v4", trans(visibleData->col(4)));

    bayesianNetwork.add("v5");
    bayesianNetwork.record("v5", trans(visibleData->col(5)));

    bayesianNetwork.add("v6");
    bayesianNetwork.record("v6", trans(visibleData->col(6)));
        
    bayesianNetwork.add("v7");
    bayesianNetwork.record("v7", trans(visibleData->col(7)));

    bayesianNetwork.add("v8");
    bayesianNetwork.record("v8", trans(visibleData->col(8)));

    bayesianNetwork.add("h");
    bayesianNetwork.record("h", *hiddenData);

    shared_ptr<mat> thetaGuessed = bayesianNetwork.imputeHiddenNode();

}
