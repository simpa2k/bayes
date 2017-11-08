//
// Created by Simon Olofsson on 2017-11-05.
//

#ifndef BAYES_BAYESIANNETWORK_H
#define BAYES_BAYESIANNETWORK_H


#include <armadillo>
#include <memory>

#include "../classifier/Classifier.h"
#include "../directedGraph/Graph.h"

using namespace std;
using namespace arma;

class BayesianNetwork {

    Classifier classifier;
    Graph<string, umat> graph;

public:

    explicit BayesianNetwork(shared_ptr<mt19937> engine);

    bool add(string nodeIdentifier);
    bool record(string nodeIdentifier, umat data);

};


#endif //BAYES_BAYESIANNETWORK_H
