//
// Created by simon on 2017-11-08.
//

#include <memory>
#include "../catch.h"
#include "../../network/BayesianNetwork.h"

using namespace std;

TEST_CASE("Test connect nodes", "[bayes]") {

    random_device r;
    auto engine = make_shared<mt19937>(r());

    BayesianNetwork bayesianNetwork(engine);

    bayesianNetwork.add("v1");
    bayesianNetwork.add("h");

    SECTION("Test connect nodes with no data") {

        bayesianNetwork.connect("v1", "h");
        vector<string> model = bayesianNetwork.getModel(); // ToDo: this might not be the best way to verify this.

        REQUIRE(model[0] == "v1");
        REQUIRE(model[1] == "h");
    }

    SECTION("Test connect nodes with data") {

        umat v1Data = {0, 1, 1, 0};
        bayesianNetwork.record("v1", v1Data);

        umat hData = {1, 1, 0, 0};
        bayesianNetwork.record("h", hData);

        bayesianNetwork.connect("v1", "h");
        vector<string> model = bayesianNetwork.getModel(); // ToDo: this might not be the best way to verify this.

        REQUIRE(model[0] == "v1");
        REQUIRE(model[1] == "h");
    }
}
