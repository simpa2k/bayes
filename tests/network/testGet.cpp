//
// Created by simon on 2017-11-08.
//

#include "../catch.h"
#include "../../network/BayesianNetwork.h"
#include "../utils/utils.h"

TEST_CASE("Test get data", "[bayes]") {

    random_device r;
    auto engine = make_shared<mt19937>(r());

    BayesianNetwork bayesianNetwork(engine);

    bayesianNetwork.add("v1");

    umat data = {1, 0, 1, 1};
    bayesianNetwork.record("v1", data);

    umat retrieved = bayesianNetwork.get("v1");

    compareMatrices(data, retrieved);

}

