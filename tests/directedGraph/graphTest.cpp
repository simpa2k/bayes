//
// Created by simon on 2017-04-04.
//
#include "../catch.h"
#include "armadillo"

#include "../../directedGraph/Graph.h"

TEST_CASE("Add data", "[graph]") {

    Graph<int, double> graph;
    REQUIRE(graph.add(1));

    SECTION("Can not add same data twice") {
        REQUIRE(!graph.add(1));
    }

    SECTION("Can add two nodes with different data") {
        REQUIRE(graph.add(2));
    }
}

TEST_CASE("Connect nodes", "[graph") {

    Graph<int, double> graph;
    REQUIRE(graph.add(1));
    REQUIRE(graph.add(2));

    REQUIRE(graph.connect(1, 2, 1.0));

    SECTION("Can have two edges to two different nodes") {

        REQUIRE(graph.add(3));
        REQUIRE(graph.connect(1, 3, 1.0));

    }

    SECTION("Nodes are only connected in one direction") {

        REQUIRE(graph.connect(2, 1, 1.0));
        SECTION("Connecting two nodes twice updates weight") {

            /*REQUIRE(!graph.connect(1, 2, 1.0));
            REQUIRE(!graph.connect(2, 1, 1.0));*/

        }
    }
}

TEST_CASE("Perform topological sort", "[graph]") {

    Graph<int, double> graph;

    graph.add(1);
    graph.add(2);
    graph.add(3);
    graph.add(4);
    graph.add(5);
    graph.add(6);
    graph.add(7);

    graph.connect(1, 2, 1.0);
    graph.connect(1, 3, 1.0);

    graph.connect(2, 4, 1.0);
    graph.connect(2, 5, 1.0);

    graph.connect(3, 6, 1.0);

    graph.connect(4, 3, 1.0);
    graph.connect(4, 6, 1.0);
    graph.connect(4, 7, 1.0);

    graph.connect(5, 4, 1.0);
    graph.connect(5, 7, 1.0);

    graph.connect(7, 6, 1.0);

    std::vector<int> correct;

    correct.push_back(1);
    correct.push_back(2);
    correct.push_back(5);
    correct.push_back(4);
    correct.push_back(3);
    correct.push_back(7);
    correct.push_back(6);

    std::vector<int> topologicalOrdering = graph.topologicalSort();

    REQUIRE(topologicalOrdering == correct);

    SECTION("Can perform two topological sorts in sequence") {

        topologicalOrdering = graph.topologicalSort();
        REQUIRE(topologicalOrdering == correct);

    }
}

