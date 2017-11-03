//
// Created by simon on 2017-10-29.
//

#include "catch.h"
#include <armadillo>

#include "../functions.h"

TEST_CASE("Replace all values of matrix", "[bayes]") {

    arma::umat dataVisible = {
            {1, 0, 0, 1, 1},
            {0, 1, 1, 1, 0}
    };

    arma::mat thetaHidden = {0.25, 0.75};
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

    auto thetaVisible = std::make_shared<std::vector<arma::mat>>();

    thetaVisible->push_back(e0);
    thetaVisible->push_back(e1);
    thetaVisible->push_back(e2);
    thetaVisible->push_back(e3);
    thetaVisible->push_back(e4);

    arma::mat replacedValues = replaceAllValues(dataVisible, 0, *thetaVisible);
    arma::mat correct = {
            {0.45, 0.60, 0.24, 0.87, 0.38},
            {0.55, 0.40, 0.76, 0.87, 0.62}
    };

    arma::umat evaluated = replacedValues == correct;
    evaluated.for_each([] (arma::uword& val) {
        REQUIRE(val == 1);
    });
}

