//
// Created by simon on 2017-11-08.
//

#include "../catch.h"
#include "../../network/BayesianNetwork.h"

using namespace std;

TEST_CASE("Add node", "[bayes]") {

    random_device r;
    auto engine = make_shared<mt19937>(r());

    BayesianNetwork bayesianNetwork(engine);

    REQUIRE(bayesianNetwork.add("v1"));

}

