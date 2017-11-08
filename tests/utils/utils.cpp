//
// Created by simon on 2017-11-08.
//

#include <armadillo>
#include "../catch.h"
#include "utils.h"
#include "../../classifier/functions.h"

void compareMatrices(const umat& expected, const umat& given) {

    umat evaluated = expected == given;
    evaluated.for_each([] (const uword& val) {
        REQUIRE(val == 1);
    });
}

shared_ptr<umat> gatherVisibleData(umat& hiddenData, vector<mat>& thetaVisible, mt19937& engine) {

    umat visibleData;
    for (auto &&node : thetaVisible) {
        umat nodeData = *simulateVisibleData(hiddenData, node, engine);
        visibleData = join_rows(visibleData, nodeData);
    }

    return make_shared<umat>(visibleData);
}

